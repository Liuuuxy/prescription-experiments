"""One BC training+eval run on the Can-MH prescription sandbox (official robomimic).
Usage: rmimic_run.py NAME MASK SEED [EPOCHS]"""
import json, sys, os, shutil
import torch
from robomimic.config import config_factory
import robomimic.scripts.train as T

name, mask, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 300

shutil.rmtree(f"/data/xinyua11/robomimic_runs/prescribe/out/{name}", ignore_errors=True)
c = config_factory(algo_name="bc")
c.experiment.name = name
c.experiment.validate = False
c.experiment.logging.terminal_output_to_txt = False
c.experiment.save.enabled = True
c.experiment.save.every_n_epochs = epochs
c.experiment.rollout.enabled = True
c.experiment.rollout.n = 100
c.experiment.rollout.horizon = 400
c.experiment.rollout.rate = 150 if name.startswith("D0only") else epochs
c.train.data = "/data/xinyua11/robomimic_runs/prescribe/can_mh_work.hdf5"
c.train.hdf5_filter_key = mask
c.train.output_dir = "/data/xinyua11/robomimic_runs/prescribe/out"
c.train.num_epochs = epochs
c.train.seed = seed
c.train.batch_size = 100
c.train.hdf5_cache_mode = "all"
c.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
c.observation.modalities.obs.rgb = []
c.lock()
# EQUAL-TRAJ weighting: each trajectory gets equal total optimizer weight
if name.startswith("eq_"):
    from robomimic.utils.dataset import SequenceDataset
    def _eqw_sampler(self):
        import torch as _t
        w = [1.0 / self._demo_id_to_demo_length[self._index_to_demo_id[i]] for i in range(len(self))]
        return _t.utils.data.WeightedRandomSampler(w, num_samples=len(self), replacement=True)
    SequenceDataset.get_dataset_sampler = _eqw_sampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T.train(c, device=device)
