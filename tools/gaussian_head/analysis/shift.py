import numpy as np, sys
S=sys.argv[1]; A=S+'/analysis'
t=np.load(A+'/tracked_flame_params_30.npz')
pose=np.concatenate([t['rotation'],t['neck_pose'],t['jaw_pose']],1)  # rad
ex=t['expr'][:,:50]
TR=np.arange(2324); 
def nn(feat, q):
    d=((feat[q][:,None,:]-feat[TR][None])**2).sum(-1); return np.sqrt(d.min(1))
p0=np.load(A+'/psnr_by_iter.npz'); te_t=p0['test_t']
dpose=np.degrees(nn(pose,te_t)); dexp=nn(ex,te_t)
# combined z-scored
z=dpose/np.median(dpose)+dexp/np.median(dexp)
print('NN pose dist deg: median %.2f p90 %.2f | train-train ref (5-frame gap excluded) below'%(np.median(dpose),np.percentile(dpose,90)))
# reference: train frames' NN distance to train frames >=15 frames away (temporal neighbours excluded)
ref=[]
rng=np.random.default_rng(0)
for i in rng.choice(2324,300,replace=False):
    m=np.abs(TR-i)>=30; d=np.sqrt(((pose[i]-pose[TR[m]])**2).sum(1)).min(); ref.append(np.degrees(d))
print('train self-NN pose (>=2s away) median %.2f p90 %.2f'%(np.median(ref),np.percentile(ref,90)))
yaw=np.degrees(t['rotation'][:,1])
print('yaw train med %.1f test med %.1f'%(np.median(yaw[TR]),np.median(yaw[te_t])))
for name in ['.','c1_ext1','c1c3']:
    p=np.load(A+'/'+name+'/psnr_by_iter.npz'); te=p['test_psnr'][:9]; its=p['iters'][:9]
    ok=te_t!=3318
    pk=te[:,ok].mean(1).argmax()
    print(f'\n== {name}  peak iter {its[pk]}')
    q=np.quantile(z[ok],[0,1/3,2/3,1])
    for b in range(3):
        m=ok&(z>=q[b])&(z<=q[b+1])
        print(f'  tercile {b} (near->far) n={m.sum()} posedist {np.median(dpose[m]):.1f}deg  PSNR@60k {te[0,m].mean():.2f} @peak {te[pk,m].mean():.2f} @540k {te[-1,m].mean():.2f}  drop peak->540k {te[pk,m].mean()-te[-1,m].mean():+.2f}')
    drop=te[pk,ok]-te[-1,ok]
    print('  corr(drop, distance) r=%.3f ; corr(PSNR540k, distance) r=%.3f; corr(PSNR540k,|yaw|) r=%.3f'%(np.corrcoef(drop,z[ok])[0,1],np.corrcoef(te[-1,ok],z[ok])[0,1],np.corrcoef(te[-1,ok],np.abs(yaw[te_t][ok]))[0,1]))
    # temporal: first vs last third of test
    tt=te_t[ok]; 
    for lo,hi in [(2324,2650),(2650,2990),(2990,3318)]:
        m=(tt>=lo)&(tt<hi); print(f'  t {lo}-{hi}: @peak {te[pk,ok][m].mean():.2f} @540k {te[-1,ok][m].mean():.2f}')
    tr=p['train_psnr'][:9]; print('  train PSNR peak->540k %.2f -> %.2f'%(tr[pk].mean(),tr[-1].mean()))
np.savez(S+'/report/shift.npz',z=z,dpose=dpose,dexp=dexp,te_t=te_t)
