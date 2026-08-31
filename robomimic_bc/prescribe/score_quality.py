"""Label-free demo-quality scoring on the W (not-yet-collected) demos.
Signals, all computed WITHOUT operator labels, at BC policies trained on D0 only:
  loss(d)  : mean BC MSE of the demo under the D0 policy
  gnorm(d) : per-demo gradient norm of that loss w.r.t. the policy nets
  length(d): trajectory length (the trivial feature)
Validation: AUC separating worse-operator vs better-operator demos in W.
This is the robomimic-domain test of the Q6 gradient quality gate."""
import json, glob
import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils

P = "/data/xinyua11/robomimic_runs/prescribe"
dev = "cuda" if torch.cuda.is_available() else "cpu"

with h5py.File(f"{P}/can_mh_work.hdf5", "r") as f:
    D0 = set(x.decode() for x in f["mask/D0"][:])
    groups = {g: set(x.decode() for x in f[f"mask/{g}"][:]) for g in ("better", "okay", "worse")}
    demos = [d for d in f["data"].keys() if d not in D0]
    obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    data = {}
    for d in demos:
        g = f["data"][d]
        obs = np.concatenate([g["obs"][k][:] for k in obs_keys], axis=1).astype(np.float32)
        act = g["actions"][:].astype(np.float32)
        data[d] = (obs, act)

def label(d):
    return "better" if d in groups["better"] else "okay" if d in groups["okay"] else "worse"

results = {}
for seed_run in ("D0only_s0", "D0only_s1", "D0only_s2"):
    ckpts = sorted(glob.glob(f"{P}/out/{seed_run}/*/models/model_epoch_300_*.pth"))
    if not ckpts: continue
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpts[-1], device=dev, verbose=False)
    net = policy.policy.nets["policy"]
    for d, (obs, act) in data.items():
        o = torch.from_numpy(obs).to(dev); a = torch.from_numpy(act).to(dev)
        od = {"robot0_eef_pos": o[:, :3], "robot0_eef_quat": o[:, 3:7],
              "robot0_gripper_qpos": o[:, 7:9], "object": o[:, 9:]}
        net.zero_grad(set_to_none=True)
        pred = net(obs_dict=od)
        loss = torch.nn.functional.mse_loss(pred, a)
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None)).item()
        results.setdefault(d, {"len": len(act), "label": label(d), "loss": [], "gnorm": []})
        results[d]["loss"].append(float(loss)); results[d]["gnorm"].append(gn)

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

W_lbl = np.array([results[d]["label"] for d in demos])
print(f"W demos: {len(demos)} | better {sum(W_lbl=='better')} okay {sum(W_lbl=='okay')} worse {sum(W_lbl=='worse')}")
for sig in ("loss", "gnorm", "len"):
    v = np.array([np.mean(results[d][sig]) if sig != "len" else results[d]["len"] for d in demos])
    a_bw = auc(v[W_lbl == "worse"], v[W_lbl == "better"])
    a_rw = auc(v[W_lbl == "worse"], v[W_lbl != "worse"])
    print(f"  {sig:6s}: AUC(worse > better) = {a_bw:.3f}   AUC(worse > rest) = {a_rw:.3f}")
json.dump({d: {k: (np.mean(v) if isinstance(v, list) else v) for k, v in r.items()}
           for d, r in results.items()}, open(f"{P}/quality_scores.json", "w"), indent=1)
print("wrote quality_scores.json")
