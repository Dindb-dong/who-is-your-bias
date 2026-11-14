"""Identity verification + head pose + expression + framing + perceptual hash for each candidate image.
usage: analyze.py <raw_dir> <candidates.json> <out.json> <ref_title_substr>[,<ref2>...]
"""
import sys, os, json, math
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "models")
RAW, CANDS, OUT, REFS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")

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


cands = json.load(open(CANDS, encoding="utf-8"))
# --- reference identity
ref = []
for c in cands:
    if any(r in c["title"] for r in REFS):
        img = imread(os.path.join(RAW, c["file"]))
        fs = faces(img)
        if len(fs):
            big = max(fs, key=lambda f: f[2] * f[3])
            ref.append(embed(img, big))
print("reference faces:", len(ref), flush=True)
ref = np.mean(ref, 0)

out = []
for c in cands:
    p = os.path.join(RAW, c["file"])
    if not os.path.exists(p):
        continue
    img = imread(p)
    if img is None:
        print("unreadable", c["file"]); continue
    H, W = img.shape[:2]
    rec_ = dict(c, width=W, height=H, n_faces=0, match=None)
    rec_["dhash"] = format(dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), "016x")
    fs = faces(img)
    rec_["n_faces"] = len(fs)
    best = None
    for f in fs:
        s = cos(embed(img, f), ref)
        if best is None or s > best[0]:
            best = (s, f)
    if best is not None:
        s, f = best
        x, y, w, h = [float(v) for v in f[:4]]
        rec_["sim"] = s
        rec_["face_bbox"] = [int(x), int(y), int(w), int(h)]
        rec_["face_px"] = int(max(w, h))
        # crop with margin for landmarker
        cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 2.0
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = img[y0:y1, x0:x1]
        rec_["crop_box"] = [x0, y0, x1, y1]
        hp = head_pose(crop)
        if hp:
            rec_.update(yaw=hp["yaw"], pitch=hp["pitch"], roll=hp["roll"], expression=expression(hp["bs"]),
                        smile=(hp["bs"].get("mouthSmileLeft", 0) + hp["bs"].get("mouthSmileRight", 0)) / 2,
                        jaw=hp["bs"].get("jawOpen", 0))
        fc = img[max(0,int(y)):max(0,int(y + h)), max(0,int(x)):max(0,int(x + w))]
        fg = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY) if fc.size else fc
        if fg.size:
            rec_["face_sharp"] = float(cv2.Laplacian(cv2.resize(fg, (160, 160)), cv2.CV_64F).var())
            rec_["face_dhash"] = format(dhash(fg), "016x")
        # body framing: pose whose nose falls inside matched face box
        sc = min(1.0, 1280 / max(H, W))
        small = cv2.resize(img, (int(W * sc), int(H * sc))) if sc < 1 else img
        pr = plm.detect(to_mp(small))
        frame = "face_only"
        for lm in pr.pose_landmarks:
            nx, ny = lm[0].x * W, lm[0].y * H
            if x - w * 0.3 <= nx <= x + w * 1.3 and y - h * 0.3 <= ny <= y + h * 1.3:
                vis = lambda i: lm[i].visibility > 0.5 and 0 <= lm[i].y <= 1.0 and 0 <= lm[i].x <= 1.0
                if (vis(27) or vis(28)) and (vis(23) or vis(24)):
                    frame = "full_body"
                elif vis(11) and vis(12) and (vis(23) or vis(24) or vis(13) or vis(14)):
                    frame = "upper_body"
                elif vis(11) or vis(12):
                    frame = "head_shoulders"
                break
        rec_["framing"] = frame
    out.append(rec_)
    print(c["file"], W, H, len(fs), round(rec_.get("sim", -1), 3), rec_.get("framing"),
          round(rec_.get("yaw", 999)), round(rec_.get("pitch", 999)), rec_.get("expression"), flush=True)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", len(out))
