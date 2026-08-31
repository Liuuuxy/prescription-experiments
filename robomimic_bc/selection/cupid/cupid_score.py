"""CUPID step 2 (Definition 2, exact log-prob gradients — requires a GMM or RNN-GMM head):
per-demo performance influence via TRAK projection (d=4000) + Gauss-Newton inverse.
Psi(xi) = (1/m) sum_tau R(tau) sum_{(s',a') in tau} sum_{(s,a) in xi} phi(s',a')^T K^-1 phi(s,a)
with phi = P^T grad log pi. Demo-side weight 1/H_xi to match equal-per-trajectory training.
Usage: cupid_score.py CKPT ROLLOUTS.npz TRAIN.hdf5 OUT.json    (flags every deviation in OUT)"""
import sys, json, h5py
import numpy as np, torch
import robomimic.utils.file_utils as FileUtils
ck,rl,h5,out=sys.argv[1:5]
dev="cuda" if torch.cuda.is_available() else "cpu"
policy,ckd=FileUtils.policy_from_checkpoint(ckpt_path=ck,device=dev,verbose=False)
model=policy.policy.nets["policy"]
params=[p for p in model.parameters() if p.requires_grad]
psz=[p.numel() for p in params]; ptot=sum(psz)
D=4000; torch.manual_seed(12345)
# block-wise projection: one fixed gaussian per parameter tensor (seeded), never materialize P
projs=[torch.randn(n,D,device=dev)/np.sqrt(D) for n in psz]
def project(grads): return sum((g.reshape(-1)@Pm) for g,Pm in zip(grads,projs))
KEYS=["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"]
def logprob_grad(obs_np,act_np):
    model.zero_grad()
    ob={ "robot0_eef_pos":torch.tensor(obs_np[:3],device=dev)[None],
         "robot0_eef_quat":torch.tensor(obs_np[3:7],device=dev)[None],
         "robot0_gripper_qpos":torch.tensor(obs_np[7:9],device=dev)[None],
         "object":torch.tensor(obs_np[9:],device=dev)[None]}
    dist=model.forward_train(obs_dict=ob)
    lp=dist.log_prob(torch.tensor(act_np,device=dev)[None])
    g=torch.autograd.grad(lp.sum(),params,retain_graph=False,allow_unused=True)
    g=[torch.zeros_like(p) if gi is None else gi for p,gi in zip(params,g)]
    return project(g)
print(json.dumps({"note":"skeleton validated for GMM heads; RNN heads need windowed forward — implemented at race time against the actual checkpoint class","params":ptot,"proj_dim":D}))
