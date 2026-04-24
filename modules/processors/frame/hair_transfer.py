# Full Hair Swap using BiSeNet Face Parsing
# Class 17 = hair in the 19-class CelebAMask-HQ label set.

import cv2
import numpy as np
import os
import onnxruntime
import modules.globals
from modules.typing import Face, Frame

HAIR_CLASS    = 17
MODEL_INPUT   = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Caches ─────────────────────────────────────────────────────────────────────
_session: onnxruntime.InferenceSession | None = None
_input_name:  str | None = None
_output_name: str | None = None

_src_cache = {'path': None, 'image': None, 'face': None, 'mask': None}


def _get_models_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', '..', 'models'))


def _get_session():
    global _session, _input_name, _output_name
    if _session is not None:
        return _session
    model_path = os.path.join(_get_models_dir(), 'bisenet_hair.onnx')
    if not os.path.exists(model_path):
        return None
    providers = getattr(modules.globals, 'execution_providers',
                        ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    _session = onnxruntime.InferenceSession(model_path, providers=providers)
    _input_name  = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    return _session


# ── BiSeNet ────────────────────────────────────────────────────────────────────

def _preprocess(frame: Frame) -> np.ndarray:
    img = cv2.resize(frame, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img.transpose(2, 0, 1), 0).astype(np.float32)


def _bisenet_hair_mask(frame: Frame, session) -> np.ndarray:
    """Return uint8 hair mask at frame resolution."""
    h, w = frame.shape[:2]
    logits  = session.run([_output_name], {_input_name: _preprocess(frame)})[0]
    classes = np.argmax(logits[0], axis=0).astype(np.uint8)
    mask512 = ((classes == HAIR_CLASS) * 255).astype(np.uint8)
    mask    = cv2.resize(mask512, (w, h), interpolation=cv2.INTER_LINEAR)
    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


# ── Alignment ──────────────────────────────────────────────────────────────────

def _build_transform(src_face: Face, tgt_face: Face) -> np.ndarray | None:
    """
    Similarity transform (scale + rotate + translate) mapping source→target,
    anchored on FOREHEAD midpoint (midway between eye-centre and top-of-bbox)
    so hair lands on the head rather than floating above it.
    """
    try:
        sk, tk = src_face.kps, tgt_face.kps
        if sk is None or tk is None:
            return None

        # Eye centres
        src_eye_c = (np.array(sk[0], np.float32) + np.array(sk[1], np.float32)) * 0.5
        tgt_eye_c = (np.array(tk[0], np.float32) + np.array(tk[1], np.float32)) * 0.5

        # Eye distance → scale
        src_ed = np.linalg.norm(np.array(sk[1], np.float32) - np.array(sk[0], np.float32))
        tgt_ed = np.linalg.norm(np.array(tk[1], np.float32) - np.array(tk[0], np.float32))
        if src_ed < 2.0 or tgt_ed < 2.0:
            return None
        scale = tgt_ed / src_ed

        # Rotation from eye line
        sa = np.arctan2(*(np.array(sk[1]) - np.array(sk[0]))[::-1])
        ta = np.arctan2(*(np.array(tk[1]) - np.array(tk[0]))[::-1])
        rot = ta - sa
        cr, sr = np.cos(rot) * scale, np.sin(rot) * scale

        # Anchor on EYE CENTRE (hair warp relative to eyes is stable)
        tx = tgt_eye_c[0] - cr * src_eye_c[0] + sr * src_eye_c[1]
        ty = tgt_eye_c[1] - sr * src_eye_c[0] - cr * src_eye_c[1]

        return np.array([[cr, -sr, tx],
                         [sr,  cr, ty]], dtype=np.float32)
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_hair_swap(target_face: Face, target_frame: Frame) -> Frame:
    if not getattr(modules.globals, 'hair_swap', False):
        return target_frame

    session = _get_session()
    if session is None:
        return target_frame

    src_path = getattr(modules.globals, 'source_path', None)
    if not src_path or not os.path.exists(src_path):
        return target_frame

    # Load + cache source
    if _src_cache['path'] != src_path:
        src_img = cv2.imread(src_path)
        if src_img is None:
            return target_frame
        try:
            from modules.face_analyser import get_one_face
            src_face = get_one_face(src_img)
        except Exception:
            return target_frame
        src_mask = _bisenet_hair_mask(src_img, session) if src_face else None
        _src_cache.update({'path': src_path, 'image': src_img,
                           'face': src_face, 'mask': src_mask})

    src_img  = _src_cache['image']
    src_face = _src_cache['face']
    src_mask = _src_cache['mask']

    if src_img is None or src_face is None or src_mask is None or target_face is None:
        return target_frame

    try:
        M = _build_transform(src_face, target_face)
        if M is None:
            return target_frame

        th, tw = target_frame.shape[:2]

        # Warp source image + hair mask into target space
        warped_src  = cv2.warpAffine(src_img,  M, (tw, th),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(src_mask, M, (tw, th),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=0)

        # Blend source hair over target — no inpainting, clean alpha blend
        alpha     = warped_mask.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)

        result = (warped_src.astype(np.float32)  * alpha_3ch +
                  target_frame.astype(np.float32) * (1.0 - alpha_3ch))
        return np.clip(result, 0, 255).astype(np.uint8)

    except Exception:
        return target_frame
