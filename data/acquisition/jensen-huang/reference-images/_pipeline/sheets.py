import json, sys, cv2, numpy as np
raw, A, outdir = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8")), sys.argv[3]
def imread(p): return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
def tile(img, label, T=240):
    h, w = img.shape[:2]; s = T / max(h, w); im = cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))))
    t = np.full((T+36, T, 3), 255, np.uint8); t[:im.shape[0], :im.shape[1]] = im
    for i, l in enumerate(label): cv2.putText(t, l, (2, T+14+i*16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
    return t
def sheet(tiles, name, cols=6):
    for k in range(0, len(tiles), 30):
        ch = tiles[k:k+30]; rows = []
        for r in range(0, len(ch), cols):
            row = ch[r:r+cols]; row += [np.full_like(ch[0], 255)] * (cols - len(row)); rows.append(np.hstack(row))
        cv2.imwrite(f"{outdir}/{name}_{k//30}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
face, full = [], []
for a in A:
    img = imread(f"{raw}/{a['file']}")
    if a.get("sim", -1) >= 0.28 and "crop_box" in a:
        x0, y0, x1, y1 = a["crop_box"]
        face.append(tile(img[y0:y1, x0:x1], [f"{a['file'][:4]} s{a['sim']:.2f} {a.get('framing','')[:9]}", f"y{a.get('yaw',999):.0f} p{a.get('pitch',999):.0f} {a.get('expression','-')}"]))
        full.append(tile(img, [f"{a['file'][:4]} {a.get('framing','')}", f"{a['width']}x{a['height']}"]))
    elif a["file"] < "c113" or a["file"] > "c167":
        full.append(tile(img, [f"{a['file'][:4]} NOMATCH s{a.get('sim',-1):.2f}", ""]))
sheet(face, "face"); sheet(full, "full")
print(len(face), len(full))
