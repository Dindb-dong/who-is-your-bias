"""Identity verification + head pose + expression + framing + perceptual hash for each candidate image.
usage: analyze.py <raw_dir> <candidates.json> <out.json> <ref_title_substr>[,<ref2>...]
"""
import sys, os, json, math
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")

det = cv2.FaceDetectorYN.create(os.path.join(M, "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(os.path.join(M, "face_recognition_sface_2021dec.onnx"), "")
flm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(M, "face_landmarker.task")),
    output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
plm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(M, "pose_landmarker_full.task")), num_poses=6))


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def faces(img):
    h, w = img.shape[:2]
    s = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
    det.setInputSize((small.shape[1], small.shape[0]))
    _, f = det.detect(small)
    if f is None:
        return []
    f = f.copy()
    f[:, :14] /= s
    return f


def embed(img, f):
    return rec.feature(rec.alignCrop(img, f)).flatten()


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dhash(gray, n=8):
    r = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA)
    return int("".join("1" if v else "0" for v in (r[:, 1:] > r[:, :-1]).flatten()), 2)


def to_mp(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))


def head_pose(crop):
    r = flm.detect(to_mp(crop))
    if not r.face_landmarks:
        return None
    R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
    # Euler (YXZ-ish): yaw around Y, pitch around X
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.asin(-max(-1, min(1, R[1, 2]))))
    roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))
    bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
    return dict(yaw=yaw, pitch=pitch, roll=roll, bs=bs)


def expression(bs):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    if smile > 0.45 and jaw > 0.25:
        return "laughing"
    if smile > 0.35:
        return "smiling"
    if jaw > 0.12:
        return "speaking"
    return "neutral"


