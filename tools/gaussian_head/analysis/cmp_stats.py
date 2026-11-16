import numpy as np, json, sys
from PIL import Image
S=sys.argv[1]; A=S+'/analysis'
cams=json.load(open(A+'/cams.json'))['cams']
def ld(p):
    im=np.asarray(Image.open(p).convert('RGBA'),dtype=np.float64)/255
    return im[...,:3]*im[...,3:]+(1-im[...,3:])
def L(p): return np.asarray(Image.open(p).convert('RGB'),dtype=np.float64)/255
def psnr(a,b,m=None):
    e=(a-b)**2; e=e.mean() if m is None else e[m].mean(); return -10*np.log10(max(e,1e-10))
rows=[]
for c in cams:
    k=f"{c['split']}_{c['timestep']:05d}"
    if c['timestep']==3318: continue
    gt,full,dc,nl=[L(f'{A}/views/{k}_{s}.png') for s in ('gt','full','dc','noLowOp')]
    wf=ld(f'{S}/cmp/filtered/{k}_webgl.png'); wa=ld(f'{S}/cmp/full/{k}_webgl.png')
    m=(np.abs(full-1).sum(-1)>0.03)|(np.abs(gt-1).sum(-1)>0.03)  # foreground (non-white)
    rows.append(dict(k=k,split=c['split'],
      full_gt=psnr(full,gt,m),dc_gt=psnr(dc,gt,m),dc_full=psnr(dc,full,m),
      wglF_dc=psnr(wf,dc,m),wglA_dc=psnr(wa,dc,m),wglF_full=psnr(wf,full,m),wglF_gt=psnr(wf,gt,m),wglA_wglF=psnr(wa,wf,m),
      mean_dc_minus_full=(dc-full)[m].mean(0).tolist(), mean_wgl_minus_dc=(wf-dc)[m].mean(0).tolist(),
      absdiff_dc_full_255=np.abs(dc-full)[m].mean()*255, absdiff_wgl_dc_255=np.abs(wf-dc)[m].mean()*255))
keys=['full_gt','dc_gt','dc_full','wglF_dc','wglA_dc','wglA_wglF','wglF_full','wglF_gt','absdiff_dc_full_255','absdiff_wgl_dc_255']
for sp in ['train','test']:
    R=[r for r in rows if r['split']==sp]
    print(sp, len(R), {k:round(float(np.mean([r[k] for r in R])),2) for k in keys})
    print('  mean RGB shift dc-full', np.round(np.mean([r['mean_dc_minus_full'] for r in R],0)*255,1), ' webgl-dc', np.round(np.mean([r['mean_wgl_minus_dc'] for r in R],0)*255,1))
# image strip for one train and one test view
for k in ['train_00664','test_02750']:
    ims=[Image.open(f'{A}/views/{k}_{s}.png').convert('RGB') for s in ('gt','full','dc')]
    for d in ('filtered','full'):
        im=Image.open(f'{S}/cmp/{d}/{k}_webgl.png'); bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im.convert('RGBA')); ims.append(bg.convert('RGB'))
    st=Image.new('RGB',(512*len(ims),512),'white')
    for i,im in enumerate(ims): st.paste(im,(512*i,0))
    st.crop((0,0,st.width,400)).resize((st.width//2,200)).save(f'{S}/report/cmp_{k}.png')
