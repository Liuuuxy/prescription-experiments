"""One benchmark run on Can-PH: BC train (equal-traj weighting, NO random-reset
rollouts) + fixed-start stratified eval on the frozen 200-state set.
Usage: rmimic_run_ph.py NAME MASK SEED [EPOCHS]
Writes out/NAME/fixed_eval.json: per-region J_r, uniform J, deployment-weighted J.
"""
import glob, json, os, shutil, sys
import numpy as np
import torch
from robomimic.config import config_factory
import robomimic.scripts.train as T

P = "/data/xinyua11/robomimic_runs/prescribe_sq"
name, mask, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 300
HORIZON = 400

shutil.rmtree(f"{P}/out/{name}", ignore_errors=True)
c = config_factory(algo_name="bc")
c.experiment.name = name
c.experiment.validate = False
c.experiment.logging.terminal_output_to_txt = False
c.experiment.save.enabled = True
c.experiment.save.every_n_epochs = epochs
c.experiment.rollout.enabled = False          # eval is ours, fixed-start, below
c.train.data = f"{P}/square_ph_work.hdf5"
c.train.hdf5_filter_key = mask
c.train.output_dir = f"{P}/out"
c.train.num_epochs = epochs
c.train.seed = seed
c.train.batch_size = 100
c.train.hdf5_cache_mode = "all"
c.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
c.observation.modalities.obs.rgb = []
c.lock()

# equal-per-trajectory weighting: ALWAYS on for this benchmark (isolates condition value)
from robomimic.utils.dataset import SequenceDataset
def _eqw_sampler(self):
    w = [1.0 / self._demo_id_to_demo_length[self._index_to_demo_id[i]] for i in range(len(self))]
    return torch.utils.data.WeightedRandomSampler(w, num_samples=len(self), replacement=True)
SequenceDataset.get_dataset_sampler = _eqw_sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T.train(c, device=device)

# ---- fixed-start stratified eval: E_probe (harvested) + sealed E_test ----
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

ckpts = glob.glob(f"{P}/out/{name}/*/models/model_epoch_{epochs}*.pth")
assert len(ckpts) == 1, f"expected 1 final ckpt, got {ckpts}"
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpts[0], device=device, verbose=False)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=f"{P}/square_ph_work.hdf5")
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                        render_offscreen=False, use_image_obs=False)
q = json.load(open(f"{P}/deploy_shares.json"))["natural_reset_shares"]
env.reset()

def eval_set(npz):
    ev = np.load(f"{P}/{npz}")
    states, regs = ev["states"], ev["region"]
    succ = []
    for i in range(len(states)):
        policy.start_episode()
        obs = env.reset_to({"states": states[i]})
        ok = False
        for t in range(HORIZON):
            obs, r, done, _ = env.step(policy(ob=obs))
            if env.is_success()["task"]:
                ok = True
                break
        succ.append(bool(ok))
    succ = np.array(succ)
    J_r = {rn: float(succ[regs == rn].mean()) for rn in sorted(set(regs.tolist()))}
    return {"name": name, "mask": mask, "seed": seed,
            "J_uniform": float(succ.mean()),
            "J_deploy": float(sum(q[rn] * J_r[rn] for rn in J_r)),
            "J_region": J_r, "n_starts": int(len(succ)),
            "successes": succ.astype(int).tolist()}

probe = eval_set("E_probe.npz")
json.dump(probe, open(f"{P}/out/{name}/fixed_eval_probe.json", "w"), indent=1)
test = eval_set("E_test.npz")
json.dump(test, open(f"{P}/out/{name}/fixed_eval_test.json", "w"), indent=1)
# stdout carries ONLY probe numbers; E_test stays sealed on disk until a
# pre-registered gate analysis reads it (see PREREG_PH_BENCHMARK.md).
print("FIXED_EVAL_PROBE", json.dumps({k: probe[k] for k in ("J_uniform", "J_deploy", "J_region")}), flush=True)
