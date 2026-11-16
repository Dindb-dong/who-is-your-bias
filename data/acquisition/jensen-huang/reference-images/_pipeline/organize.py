"""Build the final dataset folder + metadata.csv.
usage: organize.py <raw_dir> <analysis.json> <frames_dir> <frames.json> <commons.json> <dataset_dir>
"""
import sys, os, json, csv, time, shutil
import numpy as np, cv2, requests
import importlib.util
spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "an_lib.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
RAW, ANA, FDIR, FJ, CJ, DS = sys.argv[1:7]
UA = {"User-Agent": "AvatarReferenceDatasetBuilder/0.1 (student project)"}

# ---- manual review decisions (from visual inspection of contact sheets)
REJECT = {"c037": "jumbotron/screen capture", "c041": "blurry screen capture", "c104": "face occluded by 3D glasses",
          "c105": "face occluded by 3D glasses", "c168": "face occluded (hug), low light", "c169": "low-res, motion blur"}
# detector/landmarker failed on these but identity was verified visually; yaw/pitch/expression set by eye
MANUAL = {"c011": dict(yaw=40, pitch=0, expression="smiling", framing="head_shoulders"),
          "c065": dict(yaw=40, pitch=0, expression="neutral"),
          "c066": dict(yaw=-80, pitch=0, expression="smiling"),
          "c069": dict(yaw=-70, pitch=0, expression="neutral")}
MANUAL_BBOX = {"c011": None}  # whole image is a face close-up


def usage_flag(lic):
    l = lic.lower()
    if "nd" in l.split("-") or "-nd" in l:
        return "RESTRICTED: no-derivatives (reference viewing only; do not modify/redistribute)"
    if "nc" in l:
        return "LIMITED: non-commercial use only, attribution required"
    if l.startswith("cc0") or "public domain" in l:
        return "OPEN: no copyright restriction (publicity/personality rights still apply)"
    if l.startswith("cc by") or "godl" in l or "ogl" in l:
        return "OPEN: attribution required" + (" + share-alike" if "sa" in l else "")
    return "UNCLEAR: verify license before use"


def angle_name(yaw, pitch):
    a = abs(yaw)
    side = "left" if yaw < 0 else "right"
    base = "frontal" if a < 15 else (f"{side}_45" if a < 60 else f"{side}_profile")
    tilt = "_down" if pitch > 10 else ("_up" if pitch < -10 else "")
    return base, base + tilt


A = json.load(open(ANA, encoding="utf-8"))
keep = []
for a in A:
    fid = a["file"][:4]
    if fid in REJECT:
        continue
    if fid in MANUAL:
        a.update(MANUAL[fid])
        if fid in MANUAL_BBOX:
            a["face_bbox"] = [0, 0, a["width"], a["height"]]; a["face_px"] = max(a["width"], a["height"]) // 2
            a["sim"] = None
        keep.append(a); continue
    if (a.get("sim") or 0) >= 0.45 and "yaw" in a:
        keep.append(a)
print("identity-verified stills:", len(keep))

# ---- dedup by face embedding of the matched face (same photo / crop / reupload -> ~0.97+)
def face_emb(img, bbox):
    for f in an.faces(img):
        if abs(f[0] - bbox[0]) < 8 and abs(f[1] - bbox[1]) < 8:
            g = cv2.cvtColor(an.rec.alignCrop(img, f), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return an.embed(img, f), (g - g.mean()) / (g.std() + 1e-6)
    return None


def is_dup(a, b):
    if a is None or b is None:
        return False
    return an.cos(a[0], b[0]) >= 0.88 and float((a[1] * b[1]).mean()) >= 0.70

for a in keep:
    img = an.imread(os.path.join(RAW, a["file"]))
    a["_emb"] = face_emb(img, a["face_bbox"]) if a.get("sim") is not None else None
groups = []
for a in sorted(keep, key=lambda a: -(a["width"] * a["height"])):
    for g in groups:
        b = g[0]
        if is_dup(a["_emb"], b["_emb"]):
            g.append(a); break
    else:
        groups.append([a])
uniq = []
for g in groups:
    best = max(g, key=lambda a: (a["width"] * a["height"], "cropped" not in a["title"].lower()))
    best["_dups"] = [x["source_url"] for x in g if x is not best]
    uniq.append(best)
print("after dedup:", len(uniq), "| removed:", sum(len(g) - 1 for g in groups))
for g in groups:
    if len(g) > 1:
        print("  dup group:", [x["file"] + ":" + x["title"][:30] for x in g])

# ---- try original resolution for Commons images that were fetched as thumbnails
for a in uniq:
    if a["src"] == "commons" and a.get("dl_url") != a["image_url"] and a["w"] > a["width"]:
        try:
            r = requests.get(a["image_url"], headers=UA, timeout=180)
        except Exception as e:
            r = None
        if r is not None and r.status_code == 200 and len(r.content) > 1000:
            p = os.path.join(RAW, a["file"]); open(p, "wb").write(r.content)
            img = an.imread(p)
            s = img.shape[1] / a["width"]
            a["face_bbox"] = [int(v * s) for v in a["face_bbox"]]; a["face_px"] = int(a["face_px"] * s)
            a["height"], a["width"] = img.shape[:2]; a["dl_url"] = a["image_url"]
            print("  original ok", a["file"], a["width"], a["height"], flush=True)
            time.sleep(3)
        else:
            print("  original unavailable (", r.status_code if r is not None else "err", ") keep thumbnail", a["file"], flush=True)
            if r is not None and r.status_code == 429:
                break

# ---- video frames
C = {x["title"]: x for x in json.load(open(CJ, encoding="utf-8"))}
vsrc = C["File:PM Modi is such an incredible student, NVIDIA CEO Jensen Huang.webm"]
for f in json.load(open(FJ)):
    img = an.imread(os.path.join(FDIR, f["file"]))
    uniq.append(dict(f, src="commons-video", raw_dir=FDIR, title=f"{vsrc['title'][5:]} @ {f['t']:.1f}s",
                     image_url=vsrc["url"].split("?")[0], dl_url=vsrc["url"].split("?")[0], source_url=vsrc["page"],
                     license=vsrc["license"], license_url=vsrc["license_url"], author=vsrc["artist"][:120],
                     width=img.shape[1], height=img.shape[0], w=img.shape[1], h=img.shape[0],
                     framing="upper_body_frame" if not any(u.get("src") == "commons-video" for u in uniq) else "video_frame"))

# ---- write dataset
for d in ["face/frontal", "face/left_45", "face/right_45", "face/left_profile", "face/right_profile", "face/expressions",
          "upper_body", "full_body"]:
    os.makedirs(os.path.join(DS, d), exist_ok=True)
rows, n = [], 0
def write(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]); buf.tofile(path)

for a in sorted(uniq, key=lambda a: (a["src"], a["file"])):
    img = an.imread(os.path.join(a.get("raw_dir", RAW), a["file"]))
    H, W = img.shape[:2]
    x, y, w, h = a["face_bbox"]
    base, va = angle_name(a["yaw"], a["pitch"])
    face_ratio = h / H
    lic = a["license"].strip()
    common = dict(image_url=a["image_url"], downloaded_url=a.get("dl_url", a["image_url"]), source_url=a["source_url"],
                  source_site=a["src"], title=a["title"], author=a.get("author", ""), original_resolution=f"{a.get('w')}x{a.get('h')}",
                  viewing_angle=va, yaw_deg=round(a["yaw"]), pitch_deg=round(a["pitch"]), expression=a.get("expression", ""),
                  identity_score="" if a.get("sim") is None else round(a["sim"], 3),
                  license=lic, license_url=a.get("license_url", ""), usage=usage_flag(lic),
                  duplicates_removed=" | ".join(a.get("_dups", [])))
    n += 1; sid = f"JH{n:04d}"
    closeup = (w * h) / (W * H) >= 0.06
    framing = a.get("framing", "")
    # 1) the source image itself
    if closeup:
        cat = f"face/{base}"
    elif framing == "full_body":
        cat = "full_body"
    elif framing in ("upper_body", "upper_body_frame"):
        cat = "upper_body"
    else:
        cat = None  # group / wide scene: only the derived face crop is kept as a file
    if cat:
        p = f"{cat}/{sid}.jpg"; write(os.path.join(DS, p), img)
        rows.append(dict(image_id=sid, file_path=p, parent_image_id="", category=cat.split("/")[0] if "/" in cat else cat,
                         resolution=f"{W}x{H}", face_bbox_xywh=f"{x},{y},{w},{h}", face_size_px=max(w, h),
                         notes="close-up" if closeup else framing, **common))
    # 2) derived face crop (for non-close-ups with a usable face size; never for no-derivatives licenses)
    nd = "no-derivatives" in common["usage"]
    if not closeup and max(w, h) >= 70 and not nd:
        cx, cy, side = x + w / 2, y + h * 0.45, max(w, h) * 2.2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = img[y0:y1, x0:x1]
        cid = f"{sid}_face"; p = f"face/{base}/{cid}.jpg"; write(os.path.join(DS, p), crop)
        rows.append(dict(image_id=cid, file_path=p, parent_image_id=sid if cat else "", category="face",
                         resolution=f"{crop.shape[1]}x{crop.shape[0]}", face_bbox_xywh=f"{int(x-x0)},{int(y-y0)},{w},{h}",
                         face_size_px=max(w, h), notes=f"face crop of source image (crop box {x0},{y0},{x1},{y1})", **common))
    # 3) expressions folder: copies of non-neutral face images
    face_rows = [r for r in rows[-2:] if r["file_path"].startswith("face/") and r["image_id"].startswith(sid)]
    if a.get("expression") in ("smiling", "laughing", "speaking") and face_rows:
        r0 = face_rows[-1]; src = os.path.join(DS, r0["file_path"])
        p = f"face/expressions/{a['expression']}_{r0['image_id']}.jpg"; shutil.copyfile(src, os.path.join(DS, p))
        rows.append(dict(r0, image_id=f"{r0['image_id']}_expr", file_path=p, parent_image_id=r0["image_id"],
                         category="face/expressions", notes="copy for expression browsing"))

cols = ["image_id", "file_path", "category", "viewing_angle", "yaw_deg", "pitch_deg", "expression", "resolution",
        "original_resolution", "face_bbox_xywh", "face_size_px", "identity_score", "image_url", "downloaded_url", "source_url",
        "source_site", "title", "author", "license", "license_url", "usage", "parent_image_id", "duplicates_removed", "notes"]
with open(os.path.join(DS, "metadata.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
import collections
print("files:", len(rows), "| unique source images:", n)
print(collections.Counter(r["file_path"].rsplit("/", 1)[0] for r in rows))
print(collections.Counter(r["usage"].split(":")[0] for r in rows if not r["image_id"].endswith(("_face", "_expr"))))
