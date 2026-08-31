"""Generate the frozen evaluation set for the Can-PH region benchmark.

Amendment 2 (advisor): eval states come from FRESH env resets, not from the 200
demos; unique starts only (deterministic policy + env => repeats add nothing).
Collect 60 per region (50 frozen + 10 spares), stratified by the DATASET grid in
regions.json. Validates reset_to() exactness and env determinism before saving.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils

WORK = "/data/xinyua11/robomimic_runs/prescribe_ph/can_ph_work.hdf5"
OUT = "/data/xinyua11/robomimic_runs/prescribe_ph/eval_states.npz"
meta = json.load(open("/data/xinyua11/robomimic_runs/prescribe_ph/regions.json"))
QX, QY, REGIONS = meta["grid"]["qx"], meta["grid"]["qy"], meta["grid"]["labels"]
PER_REGION = 60

ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs={"obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=WORK)
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                        render_offscreen=False, use_image_obs=False)
adim = env.action_dimension
print("env ok, action dim", adim, flush=True)

def region_of_xy(x, y):
    return REGIONS[2 * int(x > QX) + int(y > QY)]

# ---- validation gate (must pass before we trust anything) ----
np.random.seed(999)
obs0 = env.reset()
st0 = env.get_state()
obs1 = env.reset_to(st0)
assert np.allclose(obs0["object"][:3], obs1["object"][:3], atol=1e-7), "reset_to changed can pos"
assert np.allclose(env.get_state()["states"], st0["states"], atol=1e-7), "state not restored exactly"
def roll_fingerprint():
    env.reset_to(st0)
    fp = []
    for t in range(20):
        o, _, _, _ = env.step(np.zeros(adim))
        fp.append(np.concatenate([o["object"], o["robot0_eef_pos"]]))
    return np.array(fp)
fa, fb = roll_fingerprint(), roll_fingerprint()
dmax = np.abs(fa - fb).max()
assert dmax == 0.0, f"env not deterministic under reset_to: max diff {dmax}"
print("VALIDATION PASS: exact restore + bitwise-deterministic 20-step replay", flush=True)

# ---- stratified collection from fresh resets ----
buckets = {r: [] for r in REGIONS}
xy = {r: [] for r in REGIONS}
n_resets, shares = 0, []
while min(len(b) for b in buckets.values()) < PER_REGION:
    np.random.seed(100000 + n_resets)          # reproducible reset stream
    obs = env.reset()
    n_resets += 1
    x, y = float(obs["object"][0]), float(obs["object"][1])
    r = region_of_xy(x, y)
    shares.append(r)
    if len(buckets[r]) < PER_REGION:
        buckets[r].append(env.get_state()["states"].copy())
        xy[r].append((x, y))

from collections import Counter
share = Counter(shares)
print(f"collected {PER_REGION}/region from {n_resets} fresh resets; "
      f"natural shares: {dict(share)}", flush=True)

states = np.stack([s for r in REGIONS for s in buckets[r]])
labels = np.array([r for r in REGIONS for _ in range(PER_REGION)], dtype="U16")
canxy = np.array([p for r in REGIONS for p in xy[r]])
np.savez_compressed(OUT, states=states, region=labels, can_xy=canxy,
                    qx=QX, qy=QY, per_region=PER_REGION)
print("saved", OUT, states.shape, flush=True)
# natural deployment shares (for the deployment-weighted objective q_r)
json.dump({"natural_reset_shares": {r: share[r] / n_resets for r in REGIONS},
           "n_resets": n_resets},
          open("/data/xinyua11/robomimic_runs/prescribe_ph/deploy_shares.json", "w"), indent=1)
