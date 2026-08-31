"""Train a clean reference BC policy on D0_bo (D0 minus worse-op demos), no rollouts."""
import sys, shutil
import torch
from robomimic.config import config_factory
import robomimic.scripts.train as T
seed = int(sys.argv[1])
name = f"ref_bo_s{seed}"
shutil.rmtree(f"/data/xinyua11/robomimic_runs/prescribe/out/{name}", ignore_errors=True)
c = config_factory(algo_name="bc")
c.experiment.name = name
c.experiment.validate = False
c.experiment.render_video = False
c.experiment.rollout.enabled = False
c.experiment.save.every_n_epochs = 300
c.train.data = "/data/xinyua11/robomimic_runs/prescribe/can_mh_work.hdf5"
c.train.hdf5_filter_key = "D0_bo"
c.train.output_dir = "/data/xinyua11/robomimic_runs/prescribe/out"
c.train.num_epochs = 300
c.train.seed = seed
c.train.batch_size = 100
c.train.hdf5_cache_mode = "all"
c.observation.modalities.obs.low_dim = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
c.observation.modalities.obs.rgb = []
c.lock()
T.train(c, device=torch.device("cuda"))
