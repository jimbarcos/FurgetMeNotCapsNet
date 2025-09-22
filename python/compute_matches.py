import os
import sys
import json
import base64
from typing import List, Tuple, Dict, Any, Optional

# Heavy deps imported lazily where possible


def _read_image_rgb(path: str):
	from PIL import Image
	img = Image.open(path).convert('RGB')
	return img


def _img_to_thumb_base64(path: str, max_side: int = 320) -> str:
	try:
		from PIL import Image
		import io
		img = Image.open(path).convert('RGB')
		img.thumbnail((max_side, max_side))
		buf = io.BytesIO()
		img.save(buf, format='JPEG', quality=85)
		return base64.b64encode(buf.getvalue()).decode('utf-8')
	except Exception:
		return ''


def _list_gallery_images(preprocessed_dir: str, pet_type: str, debug: List[str]) -> List[str]:
	want_cats = False
	want_dogs = False
	want_unknown = False
	s = (pet_type or '').lower()
	if 'cat' in s:
		want_cats = True
	if 'dog' in s:
		want_dogs = True
	if not (want_cats or want_dogs):
		want_unknown = True
		
	subdirs = []
	if want_cats:
		subdirs.append(os.path.join(preprocessed_dir, 'Cats'))
	if want_dogs:
		subdirs.append(os.path.join(preprocessed_dir, 'Dogs'))
	if want_unknown:
		subdirs.append(os.path.join(preprocessed_dir, 'Unknown'))

	exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
	files: List[str] = []
	for d in subdirs:
		if os.path.isdir(d):
			for name in os.listdir(d):
				if name.lower().endswith(exts):
					files.append(os.path.abspath(os.path.join(d, name)))
	debug.append(f"GALLERY_COUNT:{len(files)} from subdirs={subdirs}")
	return files


def _preprocess_tf(img, size: Tuple[int, int] = (224, 224)):
	import numpy as np
	from PIL import Image
	from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
	if isinstance(img, Image.Image):
		im = img
	else:
		im = Image.fromarray(img)
	im = im.resize(size, Image.BILINEAR)
	x = np.asarray(im, dtype=np.float32)
	x = preprocess_input(x)  # MobileNetV2 preprocessing
	return x


def _build_default_tf_model(debug: List[str]):
	import tensorflow as tf
	base = tf.keras.applications.MobileNetV2(
		include_top=False, weights='imagenet', input_shape=(224, 224, 3)
	)
	x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
	x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='l2_norm')(x)
	model = tf.keras.Model(inputs=base.input, outputs=x)
	debug.append('TF_MODEL:MobileNetV2+GAP+L2 (imagenet)')
	return model


def _load_tf_model(weights_path: str, attempts: List[str], debug: List[str]):
	"""Try multiple strategies to load the user's weights. Returns (model, used_tf)."""
	try:
		import tensorflow as tf  # noqa: F401
	except Exception as e:
		attempts.append(f"tensorflow_import_failed:{e}")
		return None, False

	model = None
	used_tf = True
	# Strategy 1: load_model (in case it's a full saved model)
	try:
		from tensorflow.keras.models import load_model
		# No custom_objects available in this project; try plain
		model = load_model(weights_path)
		attempts.append('load_model:success')
		debug.append('Loaded full model via load_model')
		return model, used_tf
	except Exception as e:
		attempts.append(f'load_model:failed:{e}')

	# Strategy 2: build default backbone and load weights (if compatible)
	try:
		model = _build_default_tf_model(debug)
		model.load_weights(weights_path)
		attempts.append('load_weights_on_default:success')
		debug.append('Loaded weights onto MobileNetV2 head')
		return model, used_tf
	except Exception as e:
		attempts.append(f'load_weights_on_default:failed:{e}')
		# As last resort, return the default imagenet model (without custom weights)
		try:
			model = _build_default_tf_model(debug)
			attempts.append('fallback_imagenet_backbone:success')
			return model, used_tf
		except Exception as e2:
			attempts.append(f'fallback_imagenet_backbone:failed:{e2}')
			return None, False


def _build_siamese_mnv2_tf(alpha: float, debug: List[str]):
	import tensorflow as tf
	base = tf.keras.applications.MobileNetV2(
		include_top=False, weights=None, input_shape=(224, 224, 3), alpha=alpha
	)
	x = tf.keras.layers.GlobalAveragePooling2D(name='emb_gap')(base.output)
	x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name='emb_l2')(x)
	embed = tf.keras.Model(inputs=base.input, outputs=x, name=f'embed_mnv2_a{alpha}')

	# Siamese inputs
	i1 = tf.keras.Input(shape=(224, 224, 3), name='input_a')
	i2 = tf.keras.Input(shape=(224, 224, 3), name='input_b')
	e1 = embed(i1)
	e2 = embed(i2)
	diff = tf.keras.layers.Lambda(lambda t: tf.math.abs(t[0] - t[1]), name='abs_diff')([e1, e2])
	# Simple head producing a distance-like score in [0,1]
	d = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(diff)
	out = tf.keras.layers.Dense(1, activation='sigmoid', name='distance')(d)
	model = tf.keras.Model([i1, i2], out, name='siamese_mnv2')
	debug.append(f'TF_SIAMESE:MobileNetV2 alpha={alpha}')
	return model


def _load_siamese_tf_model(weights_path: str, attempts: List[str], debug: List[str]):
	"""Build a Siamese MobileNetV2 model and try loading weights by_name.
	Returns (model or None). Only returns a model if some weights could be loaded.
	"""
	try:
		import tensorflow as tf  # noqa: F401
	except Exception as e:
		attempts.append(f"tensorflow_import_failed:{e}")
		return None

	# Try a few plausible MobileNetV2 alpha values
	for alpha in (1.0, 0.75):
		try:
			m = _build_siamese_mnv2_tf(alpha, debug)
			# Build the model by a dummy forward pass
			import numpy as np
			dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
			_ = m.predict([dummy, dummy], verbose=0)
			try:
				before = [w.numpy().copy() for w in m.weights]
				m.load_weights(weights_path, by_name=True, skip_mismatch=True)
				after = [w.numpy() for w in m.weights]
				# Heuristic: confirm at least one tensor changed
				any_changed = any((b.shape == a.shape and (b != a).any()) for b, a in zip(before, after))
				if any_changed:
					attempts.append(f'siamese_load_weights_by_name:success:alpha={alpha}')
					return m
				else:
					attempts.append(f'siamese_load_weights_by_name:no_effect:alpha={alpha}')
			except Exception as e2:
				attempts.append(f'siamese_load_weights_by_name:failed:alpha={alpha}:{e2}')
		except Exception as e:
			attempts.append(f'siamese_build_failed:alpha={alpha}:{e}')
	# Do not return an untrained model to avoid 50% constant scores
	attempts.append('siamese_no_valid_weights:return_none')
	return None


def _pairwise_similarity_tf_siamese(query_path: str, gallery_paths: List[str], top_k: int, attempts: List[str], debug: List[str]):
	"""Use a Siamese model to compute distance(query, gallery) -> similarity.
	Returns list or None on failure.
	"""
	try:
		import numpy as np
		model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'final_best_model.weights.h5'))
		m = _load_siamese_tf_model(model_path, attempts, debug)
		if m is None:
			return None

		# Preprocess images
		from PIL import Image
		def prep(p):
			img = Image.open(p).convert('RGB')
			return _preprocess_tf(img)

		q_arr = prep(query_path)
		# Batch vectorization: repeat query to match gallery batch
		Xs = []
		Ys = []
		kept_paths = []
		for p in gallery_paths:
			try:
				g_arr = prep(p)
				Xs.append(q_arr)
				Ys.append(g_arr)
				kept_paths.append(p)
			except Exception:
				continue
		if not Xs:
			return []
		X = np.stack(Xs, axis=0)
		Y = np.stack(Ys, axis=0)
		# Predict distances in [0,1] (lower is more similar)
		dists = m.predict([X, Y], batch_size=32, verbose=0).reshape(-1)
		# Detect degenerate near-constant outputs (e.g., untrained heads)
		if dists.size >= 4:
			span = float(np.ptp(dists))
			stdv = float(np.std(dists))
			mean = float(np.mean(dists))
			if span < 0.05 or stdv < 0.02 or 0.45 < mean < 0.55:
				attempts.append(f'siamese_constant_output:span={span:.4f},std={stdv:.4f},mean={mean:.4f}')
				return None
		# Convert to similarity percent
		sims = np.clip(1.0 - dists, 0.0, 1.0) * 100.0
		order = np.argsort(-sims)
		top_k = min(top_k, len(order))
		out = []
		for rank_idx in range(top_k):
			j = int(order[rank_idx])
			score = float(sims[j])
			pth = kept_paths[j]
			out.append({
				'rank': rank_idx + 1,
				'score': round(score, 2),
				'path': pth,
				'thumb_base64': _img_to_thumb_base64(pth)
			})
		return out
	except Exception as e:
		attempts.append(f'siamese_similarity_failed:{e}')
		return None


def _embeddings_tf(model, paths: List[str], batch_size: int, debug: List[str]):
	import numpy as np
	xs = []
	imgs: List[Any] = []
	for p in paths:
		try:
			img = _read_image_rgb(p)
			x = _preprocess_tf(img)
			xs.append(x)
		except Exception:
			xs.append(None)
		imgs.append(p)
	# Filter out Nones
	valid = [(p, x) for p, x in zip(imgs, xs) if x is not None]
	if not valid:
		return [], np.zeros((0,))
	imgs2, xs2 = zip(*valid)
	import numpy as np
	X = np.stack(xs2, axis=0)
	embs = []
	# Predict in batches
	for i in range(0, len(X), batch_size):
		e = model.predict(X[i:i+batch_size], verbose=0)
		embs.append(e)
	embs = np.vstack(embs)
	return list(imgs2), embs


def _cosine_to_percent(sim: float) -> float:
	# map [-1,1] -> [0,100]
	return max(0.0, min(100.0, (sim + 1.0) * 50.0))


def _similarity_tf(query_path: str, gallery_paths: List[str], top_k: int, attempts: List[str], debug: List[str]):
	try:
		import numpy as np
		model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'final_best_model.weights.h5'))
		model, used_tf = _load_tf_model(model_path, attempts, debug)
		if model is None:
			return None

		# Query embedding
		q_img = _read_image_rgb(query_path)
		q_x = _preprocess_tf(q_img)
		q_x = q_x[None, ...]
		q_emb = model.predict(q_x, verbose=0)
		# Gallery embeddings
		gal_paths, gal_embs = _embeddings_tf(model, gallery_paths, batch_size=32, debug=debug)
		if gal_embs.shape[0] == 0:
			return []

		# Cosine similarity with L2-normalized outputs (if model ensures it). Otherwise compute explicitly.
		# Normalize if not already L2
		def l2n(a, axis=1, eps=1e-9):
			n = np.linalg.norm(a, axis=axis, keepdims=True)
			return a / (n + eps)

		qn = l2n(q_emb)
		gn = l2n(gal_embs)
		sims = (qn @ gn.T).flatten()  # [-1,1]
		order = np.argsort(-sims)
		top_k = min(top_k, len(order))
		out = []
		for rank_idx in range(top_k):
			j = int(order[rank_idx])
			sim = float(sims[j])
			score = _cosine_to_percent(sim)
			pth = gal_paths[j]
			out.append({
				'rank': rank_idx + 1,
				'score': round(score, 2),
				'path': pth,
				'thumb_base64': _img_to_thumb_base64(pth)
			})
		return out
	except Exception as e:
		attempts.append(f'similarity_tf_failed:{e}')
		return None




def main():
	# Args: <image_path> <pet_type> <preprocessed_dir> <top_k> [--debug]
	try:
		if len(sys.argv) < 5:
			print(json.dumps({"ok": False, "error": "Usage: compute_matches.py <image_path> <pet_type> <pre_dir> <top_k> [--debug]"}))
			return
		image_path = sys.argv[1]
		pet_type = sys.argv[2]
		pre_dir = sys.argv[3]
		try:
			top_k = int(sys.argv[4])
		except Exception:
			top_k = 3
		want_debug = any(arg == '--debug' for arg in sys.argv[5:])

		attempts: List[str] = []
		debug: List[str] = []

		if not os.path.isfile(image_path):
			print(json.dumps({"ok": False, "error": "Query image not found"}))
			return
		if not os.path.isdir(pre_dir):
			print(json.dumps({"ok": False, "error": "Preprocessed directory not found"}))
			return

		gallery_paths = _list_gallery_images(pre_dir, pet_type, debug)
		if not gallery_paths:
			print(json.dumps({"ok": True, "matches": [], "debug": debug, "attempts": attempts}))
			return

		# 1) Try Siamese MobileNetV2 weights-by-name path
		matches = _pairwise_similarity_tf_siamese(image_path, gallery_paths, top_k, attempts, debug)
		# 2) Fall back to embedding-based MobileNetV2 (imagenet or loaded)
		if matches is None or len(matches) == 0:
			attempts.append('fallback_to_embedding_similarity')
			matches = _similarity_tf(image_path, gallery_paths, top_k, attempts, debug)

		print(json.dumps({
			"ok": True,
			"matches": matches or [],
			"debug": debug if want_debug else [],
			"attempts": attempts
		}))
	except Exception as e:
		print(json.dumps({"ok": False, "error": str(e)}))


if __name__ == '__main__':
	main()

