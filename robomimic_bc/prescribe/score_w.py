"""Label-free quality scoring of the W (not-yet-collected) demos.
Per demo: gradient norm of the BC MSE loss at each D0-trained policy (3 seeds),
plus per-demo loss. Reports AUC(worse vs better/okay) and writes scores.json."""
import glob, json
import h5py
import numpy as np
import torch

P = "/data/xinyua11/robomimic_runs/prescribe"
OBS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
import robomimic.utils.file_utils as FileUtils

ckpts = sorted(glob.glob(f"{P}/out/D0only_s*/*/models/model_epoch_300_*.pth"))
print("checkpoints:", len(ckpts))
dev = "cuda" if torch.cuda.is_available() else "cpu"

with h5py.File(f"{P}/can_mh_work.hdf5", "r") as f:
    D0 = set(x.decode() for x in f["mask/D0"][:])
    groups = {g: set(x.decode() for x in f[f"mask/{g}"][:]) for g in ("better", "okay", "worse")}
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    W = [d for d in demos if d not in D0]
    data = {}
    for d in W:
        g = f[f"data/{d}"]
        obs = np.concatenate([g[f"obs/{k}"][:] for k in OBS], axis=1).astype(np.float32)
        data[d] = (obs, g["actions"][:].astype(np.float32))

scores = {d: {"gnorm": [], "loss": []} for d in W}
for ck in ckpts:
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ck, device=dev, verbose=False)
    net = policy.policy.nets["policy"]
    net.train(False)
    params = [p for p in net.parameters() if p.requires_grad]
    for d in W:
        obs, act = data[d]
        od = {}
        i = 0
        for k, dim in zip(OBS, (3, 4, 2, 14)):
            od[k] = torch.from_numpy(obs[:, i:i+dim]).to(dev); i += dim
        target = torch.from_numpy(act).to(dev)
        pred = net(od)
        loss = torch.nn.functional.mse_loss(pred, target)
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        gn = float(torch.sqrt(sum((g**2).sum() for g in grads)))
        scores[d]["gnorm"].append(gn)
        scores[d]["loss"].append(float(loss))

out = {d: {"gnorm": float(np.mean(v["gnorm"])), "loss": float(np.mean(v["loss"])),
           "group": next(g for g, s in groups.items() if d in s)} for d, v in scores.items()}
json.dump(out, open(f"{P}/w_scores.json", "w"), indent=1)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
for key in ("gnorm", "loss"):
    w = [v[key] for v in out.values() if v["group"] == "worse"]
    bo = [v[key] for v in out.values() if v["group"] != "worse"]
    b = [v[key] for v in out.values() if v["group"] == "better"]
    print(f"{key:6s}: AUC(worse > better+okay) = {auc(w, bo):.3f}   AUC(worse > better) = {auc(w, b):.3f}")
ranked = sorted(out, key=lambda d: -out[d]["gnorm"])
bot50 = ranked[:50]   # highest gnorm = predicted worst
top50 = ranked[-50:]
frac_worse_bot = np.mean([out[d]["group"] == "worse" for d in bot50])
frac_worse_top = np.mean([out[d]["group"] == "worse" for d in top50])
print(f"predicted-worst 50: {frac_worse_bot:.0%} actually 'worse' (pool rate {len([1 for v in out.values() if v['group']=='worse'])/len(out):.0%}); predicted-best 50: {frac_worse_top:.0%}")
json.dump({"pred_worst50": bot50, "pred_best50": top50}, open(f"{P}/gradqual_sets.json", "w"))
print("wrote w_scores.json + gradqual_sets.json")
