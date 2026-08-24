"""Render one PickPlaceCounterToSink scene (fixed layout+style) as a clean
wide kitchen catalog shot. Camera: free cam aimed at the sink from the room
side; among 4 candidate 3/4 azimuths, keep the one with highest pixel variance
(most kitchen content, least flat wall) — robust to per-layout counter orientation.

Usage: render_scene.py <layout_id> <style_id> <out.png> [obj_groups]
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, robosuite, imageio, mujoco
import robocasa.environments.kitchen.atomic.kitchen_pick_place  # registers PickPlaceCounterToSink
from robosuite.controllers import load_composite_controller_config

L, Sty, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
obj = sys.argv[4] if len(sys.argv) > 4 else "cup"

cc = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(
    env_name="PickPlaceCounterToSink", robots="PandaOmron", controller_configs=cc,
    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    use_object_obs=True, ignore_done=True, seed=0, obj_groups=obj,
    obj_instance_split="pretrain", layout_and_style_ids=[[L, Sty]],
    robot_spawn_deviation_pos_x=0.0, robot_spawn_deviation_pos_y=0.0, robot_spawn_deviation_rot=0.0,
)
env.reset()
sink = np.array(env.get_fixture("sink").pos)
base = np.array(env.get_ep_meta()["init_robot_base_pos"])
v = base[:2] - sink[:2]
az0 = np.degrees(np.arctan2(v[1], v[0]))

m, d = env.sim.model._model, env.sim.data._data
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = 1   # robosuite visual geom groups
opt.geomgroup[2] = 1
ren = mujoco.Renderer(m, 384, 512)

best = None
for off in (40, -40, 60, -60, 130, -130, 150, -150):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = sink
    cam.distance = 3.5
    cam.elevation = -25
    cam.azimuth = az0 + off
    ren.update_scene(d, cam, opt)
    im = ren.render()
    # edge-density content score: mean gradient magnitude — rewards structured kitchen
    # views and rejects flat wall / wood-counter close-ups (whose pixel variance can be
    # high from grain but whose spatial structure is low).
    g = im.astype(float).mean(2)
    sc = float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
    if best is None or sc > best[0]:
        best = (sc, off, im.copy())
imageio.imwrite(out, best[2])
print(f"L{L} S{Sty} off={best[1]} std={best[0]:.1f} -> {out}")
