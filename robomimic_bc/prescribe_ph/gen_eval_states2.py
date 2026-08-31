"""E_probe / sealed E_test generation (supersedes eval_states.npz, which no
policy ever touched). Same seeded reset stream (seeds 100000+i) extended to
75/region: per region, collection order 1-25 -> E_probe, 26-75 -> E_test.
Also estimates deployment shares q_r from 2000 fresh resets.
"""
import json, os, sys
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

P = "/data/xinyua11/robomimic_runs/prescribe_ph"
meta = json.load(open(f"{P}/regions.json"))
QX, QY, REGIONS = meta["grid"]["qx"], meta["grid"]["qy"], meta["grid"]["labels"]
PER = 75  # 25 probe + 50 test
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={"obs": {"low_dim": [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=f"{P}/can_ph_work.hdf5")
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                        render_offscreen=False, use_image_obs=False)
buckets = {r: [] for r in REGIONS}
xy = {r: [] for r in REGIONS}
shares = []
i = 0
while min(len(b) for b in buckets.values()) < PER or i < 2000:
    np.random.seed(100000 + i)
    obs = env.reset()
    i += 1
    x, y = float(obs["object"][0]), float(obs["object"][1])
    r = REGIONS[2 * int(x > QX) + int(y > QY)]
    shares.append(r)
    if len(buckets[r]) < PER:
        buckets[r].append(env.get_state()["states"].copy())
        xy[r].append((x, y))
from collections import Counter
cnt = Counter(shares)
q = {r: cnt[r] / i for r in REGIONS}
print(f"{i} resets; q_r = " + " ".join(f"{r}:{q[r]:.3f}" for r in REGIONS), flush=True)

def save(fname, lo, hi):
    states = np.stack([buckets[r][k] for r in REGIONS for k in range(lo, hi)])
    labels = np.array([r for r in REGIONS for _ in range(lo, hi)], dtype="U16")
    canxy = np.array([xy[r][k] for r in REGIONS for k in range(lo, hi)])
    np.savez_compressed(f"{P}/{fname}", states=states, region=labels, can_xy=canxy,
                        qx=QX, qy=QY, per_region=hi - lo)
    print("saved", fname, states.shape, flush=True)

save("E_probe.npz", 0, 25)
save("E_test.npz", 25, 75)
json.dump({"natural_reset_shares": q, "n_resets": i},
          open(f"{P}/deploy_shares.json", "w"), indent=1)
os.remove(f"{P}/eval_states.npz")
