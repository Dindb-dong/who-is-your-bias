import sys, json, numpy as np
S=np.load('jensen_c1c3_smplx_head_b0.06.npz'); ids=np.load(sys.argv[1]).astype(np.int64)
N=len(S['v_rest']); NS=10475; vmap=np.concatenate([ids,NS+np.arange(N-NS)])
par=S['parents']; NJ=55
PD=np.zeros_like(S['pose_dirs'],dtype=np.float64); PD[vmap]=S['pose_dirs'][vmap]
def rod(a):
    th=np.linalg.norm(a)
    if th<1e-12: return np.eye(3)
    k=a/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]]); return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def fwd(pose,expr,tr):
    v=S['v_rest'].astype(np.float64)+np.einsum('e,evc->vc',expr,S['expr_dirs'].astype(np.float64))
    J=S['j_rest'].astype(np.float64)+np.einsum('e,ejc->jc',expr,S['j_expr_dirs'].astype(np.float64))
    R=np.stack([rod(p) for p in pose]); v=v+(PD.reshape(N*3,-1)@(R[1:]-np.eye(3)).reshape(-1)).reshape(N,3)
    G=np.zeros((NJ,4,4))
    for j in range(NJ):
        L=np.eye(4); L[:3,:3]=R[j]; L[:3,3]=J[j]-(J[par[j]] if j else 0); G[j]=L if j==0 else G[par[j]]@L
    A=G.copy(); A[:,:3,3]-=np.einsum('jab,jb->ja',G[:,:3,:3],J)
    T=np.einsum('vj,jab->vab',S['weights'].astype(np.float64),A); return np.einsum('vab,vb->va',T[:,:3,:3],v)+T[:,:3,3]+tr
rng=np.random.default_rng(1); cases=[]
for c in range(3):
    pose=np.zeros((NJ,3)); pose[[9,12,15,16,17,22]]=rng.normal(0,0.25,(6,3)); expr=rng.normal(0,1.0,100); tr=rng.normal(0,0.1,3)
    cases.append(dict(pose=pose.reshape(-1).tolist(),expr=expr.tolist(),trans=tr.tolist(),verts=fwd(pose,expr,tr).reshape(-1).tolist()))
json.dump(cases,open('ref_cases.json','w')); print('ref ok')
