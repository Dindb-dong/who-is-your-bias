"""Sample frames from a video, keep frames with the target identity, pick pose/expression-diverse sharp frames.
usage: video_frames.py <video> <ref_analysis.json> <raw_dir> <out_dir> <out.json>
"""
import sys, os, json
import numpy as np, cv2
args = sys.argv
import importlib.util
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "an_lib.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
VID, ANA, RAW, OUT, OJ = args[1:6]
os.makedirs(OUT, exist_ok=True)

A = json.load(open(ANA, encoding="utf-8"))
refs = []
for a in A:
    if a.get("sim", 0) > 0.8:
        img = an.imread(os.path.join(RAW, a["file"]))
        x, y, w, h = a["face_bbox"]
        for f in an.faces(img):
            if abs(f[0] - x) < 5 and abs(f[1] - y) < 5:
                refs.append(an.embed(img, f))
ref = np.mean(refs, 0)
print("refs", len(refs))

cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = int(round(fps * 0.5))
cands, i = [], 0
while True:
    ok = cap.grab()
    if not ok:
        break
    if i % step == 0:
        ok, img = cap.retrieve()
        best = None
        for f in an.faces(img):
            s = an.cos(an.embed(img, f), ref)
            if best is None or s > best[0]:
                best = (s, f)
        if best and best[0] > 0.45:
            s, f = best
            x, y, w, h = [float(v) for v in f[:4]]
            H, W = img.shape[:2]
            cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 2.0
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            hp = an.head_pose(img[y0:y1, x0:x1])
            fc = img[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)]
            sharp = float(cv2.Laplacian(cv2.resize(cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY), (160, 160)), cv2.CV_64F).var())
            if hp:
                cands.append(dict(t=i / fps, sim=s, img=img, face_bbox=[int(x), int(y), int(w), int(h)], face_px=int(max(w, h)),
                                  yaw=hp["yaw"], pitch=hp["pitch"], roll=hp["roll"], expression=an.expression(hp["bs"]), sharp=sharp))
    i += 1
print("frames with target:", len(cands))
# bin by yaw (10deg), pitch (up/level/down), expression -> sharpest per bin
bins = {}
for c in cands:
    pb = "up" if c["pitch"] > 10 else ("down" if c["pitch"] < -10 else "level")
    k = (int(c["yaw"] // 10), pb, c["expression"])
    if k not in bins or c["sharp"] > bins[k]["sharp"]:
        bins[k] = c
sel = sorted(bins.values(), key=lambda c: c["t"])
# enforce >=1.5 s gap, keep sharper
final = []
for c in sel:
    if final and c["t"] - final[-1]["t"] < 1.5:
        if c["sharp"] > final[-1]["sharp"]:
            final[-1] = c
        continue
    final.append(c)
meta = []
for c in final:
    fn = f"v_modi_{int(c['t']*10):05d}.png"
    cv2.imwrite(os.path.join(OUT, fn), c.pop("img"))
    c["file"] = fn
    meta.append(c)
    print(fn, round(c["yaw"]), round(c["pitch"]), c["expression"], round(c["sharp"]))
json.dump(meta, open(OJ, "w"), indent=1)
print("selected", len(meta))
