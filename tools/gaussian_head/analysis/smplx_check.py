import os, sys, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','bundle'))
from head_bundle import read_bundle
V=os.environ['SMPLX_ASSETS'].rstrip('/')+'/'  # SMPL-X model + FLAME vertex-id root (research license; not in git)
ids=np.load(V+'unilstalk-dyad-render-v4/flame_render/assets/SMPL-X__FLAME_vertex_ids.npy')
print('correspondence', ids.shape, ids.dtype, 'unique', len(np.unique(ids)), 'range', ids.min(), ids.max())
for p in ['smplx_repro/models/smplx/SMPLX_NEUTRAL_2020.npz','smplx_repro/models/smplx/SMPLX_NEUTRAL.npz']:
    m=np.load(V+p, allow_pickle=False)
    print('\n==',p); print({k:m[k].shape for k in m.files if k in ('v_template','f','shapedirs','posedirs','J_regressor','kintree_table','weights','lmk_faces_idx')})
    f=m['f'].astype(np.int64); vt=m['v_template']
    H,B=read_bundle('analysis/jensen_a_full'); ff=B['faces'].astype(np.int64)[:9976]
    mapped=ids[ff]
    S=set(map(tuple,np.sort(f,1)))
    hit=np.array([tuple(r) in S for r in np.sort(mapped,1)])
    # orientation check
    Sor=set(map(tuple,np.concatenate([f,np.roll(f,1,1),np.roll(f,2,1)])))
    same_or=np.array([tuple(r) in Sor for r in mapped])
    print('FLAME faces (no teeth) found in SMPL-X faces: %d / %d, same winding %d'%(hit.sum(),len(hit),same_or.sum()))
    # shape diff: FLAME template vs SMPL-X template head
    fl=B['v_shaped'][:5023]
    a=vt[ids]; print('SMPL-X head template vs FLAME v_shaped (this person): mean offset', (fl-a).mean(0).round(4), 'rms after centering %.4f m'%np.sqrt((((fl-fl.mean(0))-(a-a.mean(0)))**2).sum(1).mean()))
    sd=m['shapedirs']; print('shapedirs', sd.shape, '-> betas+expr dims', sd.shape[-1])
    gb=B['g_binding'].astype(np.int64); g=np.load('analysis/c1c3/gauss_600k.npz')['binding']
    print('teeth faces (>=9976) used by gaussians: c1c3 %d of %d'%((g>=9976).sum(),len(g)))
