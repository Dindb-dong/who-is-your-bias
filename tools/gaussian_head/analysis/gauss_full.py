import os, sys, pickle, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','bundle'))
from head_bundle import read_bundle, flame_forward, face_frames
H,B=read_bundle('analysis/jensen_a_full')
faces=B['faces'].astype(np.int64); F=len(faces)
verts=flame_forward(B,B['m_expr'][0].astype(np.float64),B['m_pose'][0].astype(np.float64),B['m_trans'][0].astype(np.float64))
fn=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-12
area=0.5*np.linalg.norm(np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]]),axis=1)
masks=pickle.load(open('flame_masks_zip/FLAME_masks.pkl','rb'),encoding='latin1')
def fr(name):
    if name=='teeth': return np.arange(F)>=9976
    vs=np.zeros(5023,bool); vs[np.asarray(masks[name])]=True
    m=np.zeros(F,bool); m[:9976]=vs[faces[:9976]].all(1); return m
regions=['face','eye_region','nose','lips','teeth','forehead','scalp','left_ear','right_ear','neck','boundary']
RM={r:fr(r) for r in regions}
for name in ['.','c1_ext1','c1c3']:
    g=np.load(f'analysis/{name}/gauss_600k.npz'); b=g['binding']; op=g['opacity']; sc=g['scale']; N=len(b)
    low=op<0.005
    cnt=np.bincount(b,minlength=F); lowcnt=np.bincount(b[low],minlength=F)
    allLow=cnt==lowcnt  # faces whose every point is low -> guard keeps them all
    guarded=np.isin(b,np.where(allLow)[0])&low
    print(f'\n===== {name}: N={N:,}  opacity<0.005: {low.sum():,} ({low.mean()*100:.1f}%)  prunable under guard: {(low&~guarded).sum():,}  kept-by-guard: {guarded.sum():,}  faces w/ gaussians {np.count_nonzero(cnt)}')
    print('  opacity pct p10/50/90', np.percentile(op,[10,50,90]).round(4), ' share<0.05', (op<0.05).mean().round(3))
    vd=g['cam_center'][None]-g['xyz']; vd/=np.linalg.norm(vd,axis=1,keepdims=True)
    ma=g['min_axis']; cos_view=np.abs((ma*vd).sum(1)); cos_n=np.abs((ma*fn[b]).sum(1))
    smin=sc.min(1); smax=sc.max(1); smid=np.median(sc,1)
    vis=op>=0.005
    print(f"  {'region':10s} {'faces':>5s} {'cm2':>6s} {'N':>8s} {'vis':>7s} {'vis/cm2':>8s} {'low%':>5s} {'medOp':>6s} {'max_mm':>7s} {'min_um':>7s} {'aniso':>8s} {'|cos(min,view)|':>15s} {'|cos(min,normal)|':>17s}")
    for r in regions:
        m=RM[r][b]; v=m&vis; A=area[RM[r]].sum()*1e4
        if m.sum()==0: continue
        print(f"  {r:10s} {RM[r].sum():5d} {A:6.1f} {m.sum():8d} {v.sum():7d} {v.sum()/A:8.1f} {low[m].mean()*100:5.1f} {np.median(op[m]):6.3f} {np.median(smax[v])*1e3:7.2f} {np.median(smin[v])*1e6:7.2f} {np.median(smax[v]/smin[v]):8.1e} {np.median(cos_view[v]):15.2f} {np.median(cos_n[v]):17.2f}")
    print('  visible: share |cos(min_axis,view)|>0.9: %.2f  (random 3D dir would be 0.10) ; >0.9 vs normal: %.2f'%((cos_view[vis]>0.9).mean(),(cos_n[vis]>0.9).mean()))
    print('  visible: min/mid/max scale medians (mm) %.4f / %.3f / %.3f'%(np.median(smin[vis])*1e3,np.median(smid[vis])*1e3,np.median(smax[vis])*1e3))
    print('  raw_log_scale min-axis p50/p5: ', np.percentile(g['raw_log_scale'].min(1)[vis],[50,5]).round(1), ' max-axis p50:', np.percentile(g['raw_log_scale'].max(1)[vis],50).round(2))
    print('  sh_rest_norm/|sh_dc| median (visible): %.2f'%np.median(g['sh_rest_norm'][vis]/(np.linalg.norm(g['sh_dc'][vis],axis=1)+1e-6)))
