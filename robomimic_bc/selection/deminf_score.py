"""DemInf-style mutual-information demo scoring on Can-MH (300 demos).
Two small beta-VAEs (state->8, action->4), then KSG-1 MI per (s,a) pair in randomized
batches; demo score = mean pair MI. Also raw-vector ablation (no VAE)."""
import h5py, json, time
import numpy as np, torch, torch.nn as nn
from scipy.special import digamma
dev="cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
H5="/data/xinyua11/robomimic_datasets/can/mh/low_dim_v141.hdf5"
KEYS=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
f=h5py.File(H5,"r")
demos=sorted(f["data"].keys(),key=lambda d:int(d.split("_")[1]))
S=[];A=[];idx=[]
for d in demos:
    g=f["data"][d]
    s=np.concatenate([g["obs"][k][:] for k in KEYS],axis=1)
    a=g["actions"][:]
    idx.append((len(S) and sum(len(x) for x in S) or 0, s.shape[0]))
    S.append(s);A.append(a)
S=np.concatenate(S).astype(np.float32); A=np.concatenate(A).astype(np.float32)
off=np.cumsum([0]+[f["data"][d].attrs["num_samples"] for d in demos])
f.close()
print(f"{len(demos)} demos, {len(S)} pairs, state {S.shape[1]}d action {A.shape[1]}d")
Sm,Ss=S.mean(0),S.std(0)+1e-6; Am,As=A.mean(0),A.std(0)+1e-6
Sw=(S-Sm)/Ss; Aw=(A-Am)/As
class VAE(nn.Module):
    def __init__(s_,din,dz):
        super().__init__()
        s_.e=nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s_.mu=nn.Linear(128,dz); s_.lv=nn.Linear(128,dz)
        s_.d=nn.Sequential(nn.Linear(dz,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,din))
    def forward(s_,x):
        h=s_.e(x); mu,lv=s_.mu(h),s_.lv(h)
        z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        return s_.d(z),mu,lv
def train_vae(X,dz,beta=0.02,epochs=50,tag=""):
    m=VAE(X.shape[1],dz).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    Xt=torch.tensor(X,device=dev)
    for ep in range(epochs):
        perm=torch.randperm(len(Xt),device=dev)
        tot=0
        for i in range(0,len(Xt),4096):
            xb=Xt[perm[i:i+4096]]
            xr,mu,lv=m(xb)
            rec=((xr-xb)**2).sum(1).mean()
            kl=(-0.5*(1+lv-mu**2-lv.exp()).sum(1)).mean()
            loss=rec+beta*kl
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
        if ep%10==0: print(f"  {tag} ep{ep} loss {tot:.1f}",flush=True)
    with torch.no_grad():
        return m.mu(m.e(Xt)).cpu().numpy()
t0=time.time()
Zs=train_vae(Sw,8,tag="stateVAE"); Za=train_vae(Aw,4,tag="actionVAE")
print(f"VAEs done {time.time()-t0:.0f}s")
def ksg_scores(zs,za,k=5,bs=1024,passes=4,seed=0):
    rng=np.random.RandomState(seed); n=len(zs)
    acc=np.zeros(n); cnt=np.zeros(n)
    for p in range(passes):
        order=rng.permutation(n)
        for i in range(0,n,bs):
            b=order[i:i+bs]
            if len(b)<k+2: continue
            xs,xa=zs[b],za[b]
            ds=np.max(np.abs(xs[:,None,:]-xs[None,:,:]),axis=2)
            da=np.max(np.abs(xa[:,None,:]-xa[None,:,:]),axis=2)
            dj=np.maximum(ds,da); np.fill_diagonal(dj,np.inf)
            eps=np.sort(dj,axis=1)[:,k-1]
            np.fill_diagonal(ds,np.inf); np.fill_diagonal(da,np.inf)
            nx=(ds<eps[:,None]).sum(1); ny=(da<eps[:,None]).sum(1)
            mi=digamma(k)+digamma(len(b))-digamma(nx+1)-digamma(ny+1)
            acc[b]+=mi; cnt[b]+=1
    return acc/np.maximum(cnt,1)
t0=time.time()
mi=ksg_scores(Zs,Za)
lo,hi=np.percentile(mi,[1,99]); mi=np.clip(mi,lo,hi)
mi_raw=ksg_scores(Sw,Aw)
lo,hi=np.percentile(mi_raw,[1,99]); mi_raw=np.clip(mi_raw,lo,hi)
print(f"KSG done {time.time()-t0:.0f}s")
dscore={}; dscore_raw={}
for j,d in enumerate(demos):
    sl=slice(off[j],off[j+1])
    dscore[d]=float(mi[sl].mean()); dscore_raw[d]=float(mi_raw[sl].mean())
json.dump({"vae":dscore,"raw":dscore_raw},open("deminf_scores.json","w"),indent=1)
from scipy import stats
v=np.array([dscore[d] for d in demos]); r=np.array([dscore_raw[d] for d in demos])
print(f"VAE-vs-raw ranking agreement: spearman {stats.spearmanr(v,r)[0]:+.3f}")
# vs operator tiers
fq=h5py.File(H5,"r")
tier={}
for t in ("better","okay","worse"):
    for d in fq["mask"][t][:]: tier[d.decode()]=t
fq.close()
def auc(pos,neg):
    a=np.array(pos);b=np.array(neg)
    return float((np.subtract.outer(a,b)>0).mean()+0.5*(np.subtract.outer(a,b)==0).mean())
for nm,sc in (("VAE",dscore),("raw",dscore_raw)):
    w=[sc[d] for d in demos if tier[d]=="worse"]; b=[sc[d] for d in demos if tier[d]=="better"]
    print(f"DemInf({nm}): AUC(better>worse by score) {auc(b,w):.3f}  (their claim: high MI = good demo)")
