"""Build internal_reference/ (copyrighted, team-internal only) from NVIDIA newsroom photos + official video frames.
usage: organize_internal.py <open_raw> <open_analysis.json> <out_dir> <raw:analysis.json>[,<raw:analysis.json>...] [<frames_dir> <frames.json> <video_meta.json>]
"""
import sys, os, json, csv, shutil, collections
import numpy as np, cv2
import importlib.util
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "an_lib.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
ORAW, OANA, DS, SETS = sys.argv[1:5]
VID = sys.argv[5:8] if len(sys.argv) >= 8 else None

USAGE = "INTERNAL: (c) NVIDIA press material - team-internal modeling reference only; do not commit, publish or redistribute"
REJECT = {"n007": "wide stage shot, face too small", "n009": "wide stage shot, face too small", "ref_": "reference copy"}
# landmarker failed on these (strong head turn); identity verified visually, yaw/pitch/expression set by eye
MANUAL = {"b008": dict(yaw=70, pitch=0, expression="smiling"), "b059": dict(yaw=-75, pitch=0, expression="neutral"),
          "b119": dict(yaw=65, pitch=0, expression="smiling"), "b156": dict(yaw=-35, pitch=25, expression="smiling"),
          "b186": dict(yaw=-65, pitch=0, expression="neutral")}


def angle_name(yaw, pitch):
    a = abs(yaw); side = "left" if yaw < 0 else "right"
    base = "frontal" if a < 15 else (f"{side}_45" if a < 60 else f"{side}_profile")
    return base, base + ("_down" if pitch > 10 else ("_up" if pitch < -10 else ""))


def face_emb(img, bbox):
    for f in an.faces(img):
        if abs(f[0] - bbox[0]) < 8 and abs(f[1] - bbox[1]) < 8:
            g = cv2.cvtColor(an.rec.alignCrop(img, f), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return an.embed(img, f), (g - g.mean()) / (g.std() + 1e-6)
    return None


def is_dup(a, b):
    return a is not None and b is not None and an.cos(a[0], b[0]) >= 0.88 and float((a[1] * b[1]).mean()) >= 0.70


# open-set embeddings (to avoid duplicating images already in the open dataset)
open_embs = []
for a in json.load(open(OANA, encoding="utf-8")):
    if (a.get("sim") or 0) >= 0.45 and "face_bbox" in a:
        e = face_emb(an.imread(os.path.join(ORAW, a["file"])), a["face_bbox"])
        if e: open_embs.append((a["source_url"], e))

keep = []
for pair in SETS.split(","):
    RAW, ANA = pair.split(":")
    for a in json.load(open(ANA, encoding="utf-8")):
        if a["file"][:4] in REJECT or ((a.get("sim") or 0) < 0.45 and a["file"][:4] not in MANUAL):
            continue
        if a["file"][:4] in MANUAL:
            a.update(MANUAL[a["file"][:4]])
        if "yaw" in a:
            a["raw_dir"] = RAW; keep.append(a)
if VID:
    fdir, fj, vm = VID
    vm = json.load(open(vm, encoding="utf-8"))
    for f in json.load(open(fj)):
        img = an.imread(os.path.join(fdir, f["file"]))
        keep.append(dict(f, raw_dir=fdir, src="nvidia-newsroom-video", title=f"{vm['title']} @ {f['t']:.1f}s",
                         image_url=vm["url"], dl_url=vm["url"], source_url=vm["page"], license=vm["license"],
                         license_url=vm.get("license_url", ""), author="NVIDIA", width=img.shape[1], height=img.shape[0],
                         w=img.shape[1], h=img.shape[0], framing="video_frame"))

for a in keep:
    a["_emb"] = face_emb(an.imread(os.path.join(a["raw_dir"], a["file"])), a["face_bbox"])
uniq, dropped = [], []
for a in sorted(keep, key=lambda a: -(a["width"] * a["height"])):
    od = next((u for u, e in open_embs if is_dup(a["_emb"], e)), None)
    if od:
        dropped.append((a["file"], "already in open set: " + od)); continue
    # video frames share one webcam scene, so frame-vs-frame similarity is always high; they were already
    # time-spaced and binned by pose/expression, so only compare them against stills
    isv = a["src"].endswith("video")
    m = next((u for u in uniq if not (isv and u["src"].endswith("video")) and is_dup(a["_emb"], u["_emb"])), None)
    if m:
        m.setdefault("_dups", []).append(a["source_url"] + "#" + a["file"]); dropped.append((a["file"], "dup of " + m["file"])); continue
    uniq.append(a)
print("kept", len(uniq), "dropped", len(dropped))
for d in dropped: print("  drop", *d)

if os.path.isdir(DS):
    shutil.rmtree(DS)
for d in ["face/frontal", "face/left_45", "face/right_45", "face/left_profile", "face/right_profile", "face/expressions",
          "upper_body", "full_body"]:
    os.makedirs(os.path.join(DS, d), exist_ok=True)


def write(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]); buf.tofile(path)


rows, n, vfirst = [], 0, True
for a in sorted(uniq, key=lambda a: (a["src"], a["file"])):
    img = an.imread(os.path.join(a["raw_dir"], a["file"])); H, W = img.shape[:2]
    x, y, w, h = a["face_bbox"]; base, va = angle_name(a["yaw"], a["pitch"])
    common = dict(image_url=a["image_url"], downloaded_url=a.get("dl_url", a["image_url"]), source_url=a["source_url"],
                  source_site=a["src"], title=a["title"], author=a.get("author", ""), original_resolution=f"{a.get('w')}x{a.get('h')}",
                  viewing_angle=va, yaw_deg=round(a["yaw"]), pitch_deg=round(a["pitch"]), expression=a.get("expression", ""),
                  identity_score=round(a["sim"], 3), license=a["license"], license_url=a.get("license_url", ""), usage=USAGE,
                  duplicates_removed=" | ".join(a.get("_dups", [])))
    n += 1; sid = f"JHI{n:04d}"
    closeup = (w * h) / (W * H) >= 0.06
    fr = a.get("framing", "")
    if closeup:
        cat = f"face/{base}"
    elif fr == "full_body":
        cat = "full_body"
    elif fr == "upper_body" or (fr == "video_frame" and vfirst):
        cat = "upper_body"; vfirst = vfirst and fr != "video_frame"
    else:
        cat = None
    if cat:
        p = f"{cat}/{sid}.jpg"; write(os.path.join(DS, p), img)
        rows.append(dict(image_id=sid, file_path=p, parent_image_id="", category=cat.split("/")[0], resolution=f"{W}x{H}",
                         face_bbox_xywh=f"{x},{y},{w},{h}", face_size_px=max(w, h), notes="close-up" if closeup else fr, **common))
    if not closeup and max(w, h) >= 70:
        cx, cy, side = x + w / 2, y + h * 0.45, max(w, h) * 2.2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)); x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = img[y0:y1, x0:x1]; cid = f"{sid}_face"; p = f"face/{base}/{cid}.jpg"; write(os.path.join(DS, p), crop)
        rows.append(dict(image_id=cid, file_path=p, parent_image_id=sid if cat else "", category="face",
                         resolution=f"{crop.shape[1]}x{crop.shape[0]}", face_bbox_xywh=f"{int(x-x0)},{int(y-y0)},{w},{h}",
                         face_size_px=max(w, h), notes=f"face crop of source image (crop box {x0},{y0},{x1},{y1})", **common))
    fr_rows = [r for r in rows[-2:] if r["file_path"].startswith("face/") and r["image_id"].startswith(sid)]
    if a.get("expression") in ("smiling", "laughing", "speaking") and fr_rows:
        r0 = fr_rows[-1]; p = f"face/expressions/{a['expression']}_{r0['image_id']}.jpg"
        shutil.copyfile(os.path.join(DS, r0["file_path"]), os.path.join(DS, p))
        rows.append(dict(r0, image_id=f"{r0['image_id']}_expr", file_path=p, parent_image_id=r0["image_id"],
                         category="face/expressions", notes="copy for expression browsing"))

cols = ["image_id", "file_path", "category", "viewing_angle", "yaw_deg", "pitch_deg", "expression", "resolution",
        "original_resolution", "face_bbox_xywh", "face_size_px", "identity_score", "image_url", "downloaded_url", "source_url",
        "source_site", "title", "author", "license", "license_url", "usage", "parent_image_id", "duplicates_removed", "notes"]
with open(os.path.join(DS, "metadata_internal.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
print("files:", len(rows), "| unique sources:", n)
print(collections.Counter(r["file_path"].rsplit("/", 1)[0] for r in rows))
