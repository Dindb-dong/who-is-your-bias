import json, requests, sys, re, html
UA = {"User-Agent": "dataset-collector/1.0 (academic research)"}
API = "https://commons.wikimedia.org/w/api.php"
def q(**p):
    p.update(format="json")
    out=[]; cont={}
    while True:
        import time
        for t in range(6):
            rr=requests.get(API, params={**p, **cont}, headers=UA, timeout=60)
            if rr.status_code==200 and rr.text.startswith("{"): break
            print("retry", rr.status_code, rr.text[:120].encode("ascii","replace")); time.sleep(5*(t+1))
        r=rr.json(); time.sleep(0.5)
        out.append(r)
        if "continue" not in r: break
        cont=r["continue"]
    return out
seen_cats=set(); files={}
def walk(cat, depth):
    if cat in seen_cats or depth>2: return
    seen_cats.add(cat)
    for r in q(action="query", list="categorymembers", cmtitle=cat, cmlimit=500):
        for m in r["query"]["categorymembers"]:
            if m["ns"]==14: walk(m["title"], depth+1)
            elif m["ns"]==6: files.setdefault(m["title"], cat)
walk("Category:Jen-Hsun Huang", 0)
# also search results
for r in q(action="query", list="search", srsearch="Jensen Huang", srnamespace=6, srlimit=500):
    for m in r["query"]["search"]: files.setdefault(m["title"], "search")
titles=list(files); res=[]
def strip(s): return html.unescape(re.sub("<[^>]+>","",s or "")).strip()
for i in range(0,len(titles),40):
    for r in q(action="query", titles="|".join(titles[i:i+40]), prop="imageinfo", iiprop="url|size|mime|extmetadata|sha1"):
        for p in r["query"]["pages"].values():
            if "imageinfo" not in p: continue
            ii=p["imageinfo"][0]; em=ii.get("extmetadata",{})
            g=lambda k: strip(em.get(k,{}).get("value",""))
            res.append(dict(title=p["title"], cat=files.get(p["title"]), url=ii["url"], page=ii["descriptionurl"],
              w=ii["width"], h=ii["height"], size=ii["size"], mime=ii["mime"], sha1=ii["sha1"],
              license=g("LicenseShortName"), license_url=g("LicenseUrl"), artist=g("Artist"), date=g("DateTimeOriginal"),
              desc=g("ImageDescription")[:200]))
json.dump(res, open(sys.argv[1],"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("cats:", len(seen_cats)); [print(" ",c.encode("ascii","replace").decode()) for c in seen_cats]
print("files:", len(res))
