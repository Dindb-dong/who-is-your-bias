import json, requests, os, time, hashlib, re, sys
UA={"User-Agent":"dataset-collector/1.0 (academic research)"}
RAW=sys.argv[1]; os.makedirs(RAW,exist_ok=True)
cands=[]
for x in json.load(open("commons.json",encoding="utf-8")):
    if x["mime"] not in ("image/jpeg","image/png") or x["size"]>40e6: continue
    if any(k in (x["cat"] or "") for k in ("Engineering Center","sundial")): continue
    if "genome" in x["title"]: continue
    cands.append(dict(src="commons", title=x["title"][5:], image_url=x["url"], source_url=x["page"], w=x["w"], h=x["h"],
        license=x["license"], license_url=x["license_url"], author=x["artist"][:120], date=x["date"][:40]))
LIC={"by":"CC BY","by-sa":"CC BY-SA","by-nc":"CC BY-NC","by-nd":"CC BY-ND","by-nc-sa":"CC BY-NC-SA","by-nc-nd":"CC BY-NC-ND","cc0":"CC0","pdm":"Public Domain Mark"}
for x in json.load(open("openverse.json",encoding="utf-8")).values():
    if x["source"]!="flickr": continue
    cands.append(dict(src="flickr(openverse)", title=x["title"] or "", image_url=x["url"], source_url=x["foreign_landing_url"], w=x.get("width"), h=x.get("height"),
        license=LIC.get(x["license"],x["license"])+" "+(x.get("license_version") or ""), license_url=x.get("license_url",""), author=(x.get("creator") or "")[:120], date=""))
out=[]
for i,c in enumerate(cands):
    ext=os.path.splitext(c["image_url"].split("?")[0])[1].lower() or ".jpg"
    fn=f"c{i:03d}{ext}"; p=os.path.join(RAW,fn)
    c["dl_url"]=c["image_url"]
    if not os.path.exists(p):
        urls=[c["image_url"]]
        if c["src"]=="commons":
            u=c["image_url"].split("?")[0]; rel=u.split("/commons/",1)[1]; name=u.rsplit("/",1)[1]
            steps=[x for x in (3840,1920,1280,960,500) if x < c["w"]]
            if not steps: print("SKIP small",fn,flush=True); continue
            urls=["https://thumb.wikimedia.org/wikipedia/commons/thumb/"+rel+"/"+str(steps[0])+"px-"+name]
        ok=False
        for u in urls:
            for t in range(4):
                try: r=requests.get(u,headers=UA,timeout=120)
                except Exception as e: print("err",e.__class__.__name__,fn,flush=True); time.sleep(20); continue
                if r.status_code==200 and len(r.content)>1000: open(p,"wb").write(r.content); ok=True; c["dl_url"]=u; break
                wait=int(r.headers.get("Retry-After","0") or 0) or 15*(t+1)
                print("retry",r.status_code,fn,"wait",wait,flush=True); time.sleep(min(wait,120))
            if ok: break
        if not ok: print("FAIL",fn,flush=True); continue
        print("ok",fn,flush=True); time.sleep(1.5)
    c["file"]=fn; out.append(c)
json.dump(out,open("candidates.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(len(cands),"candidates,",len(out),"downloaded")
