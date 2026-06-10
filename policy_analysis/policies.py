"""Pluggable policy interface for the RoboCasa policy-analysis harness.

A policy is any callable: ``action = policy(obs)`` where ``obs`` is the
robosuite observation dict (contains ``*_image`` arrays and proprio keys) and
``action`` is a 12-D numpy array in the env's action space.

To evaluate a *real* trained model (Diffusion Policy / GR00T / pi0) on the H100,
implement a small adapter that loads the checkpoint and maps RoboCasa obs ->
the model's expected input and the model's output -> a 12-D action. See
``CheckpointPolicyStub`` for the shape of that adapter.
"""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Uniform-random actions. Used to self-test the harness plumbing.

    Base motion is zeroed (matching robocasa.run_random_rollouts) to avoid
    excessive jitter from a wandering mobile base.
    """

    def __init__(self, action_low, action_high, zero_base=True, seed=None):
        self.low = np.asarray(action_low, dtype=np.float64)
        self.high = np.asarray(action_high, dtype=np.float64)
        self.zero_base = zero_base
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        a = self.rng.uniform(self.low, self.high)
        if self.zero_base:
            # action layout (see env_utils.convert_action): base_motion = a[7:11]
            a[7:11] = 0.0
        return a


class CheckpointPolicyStub:
    """Template adapter for a real trained policy. NOT used in self-tests.

    Fill in ``_load`` and ``__call__`` on the H100 where the checkpoint and the
    model's package (diffusion_policy / Isaac-GR00T / openpi) are installed.

    Typical wiring:
        * load the checkpoint + model config
        * build the obs dict the model expects (stack the camera images named
          in ``camera_names``, normalize, add history if the model is temporal)
        * run the model, denormalize, and return a 12-D action array
    """

    def __init__(self, checkpoint_path, camera_names, device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.camera_names = camera_names
        self.device = device
        self.model = self._load()

    def _load(self):
        raise NotImplementedError(
            "Implement checkpoint loading for your framework on the H100."
        )

    def reset(self):
        """Clear any temporal/history buffers between episodes."""
        pass

    def __call__(self, obs):
        raise NotImplementedError(
            "Map RoboCasa obs -> model input, run model, return 12-D action."
        )
