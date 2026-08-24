import numpy as np
import robosuite
import robocasa  # noqa
from robosuite.controllers import load_composite_controller_config

cc = load_composite_controller_config(robot='PandaOmron')
for L in [56, 40, 15, 20, 33]:
    env = robosuite.make(env_name='PickPlaceCounterToSink', robots='PandaOmron',
                         controller_configs=cc, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, use_object_obs=True, ignore_done=True, seed=0,
                         layout_and_style_ids=[[L, 11]])
    for rep in range(2):
        env.reset()
        env.step(np.zeros(env.action_dim))
        sink = env.get_fixture('sink')
        sp = np.asarray(sink.pos)[:2]
        yaw = float(np.asarray(sink.rot).ravel()[0])
        R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
        out = {}
        for body in ['mobilebase0_base', 'mobilebase0_support', 'robot0_base']:
            try:
                bp = np.array(env.sim.data.get_body_xpos(body))[:2]
                out[body] = np.round(R @ (bp - sp), 2).tolist()
            except Exception as ex:
                out[body] = repr(ex)[:40]
        try:
            eef = np.array(env.sim.data.get_site_xpos('gripper0_right_grip_site'))[:2]
            out['eef'] = np.round(R @ (eef - sp), 2).tolist()
        except Exception:
            out['eef'] = 'na'
        print('layout', L, 'rep', rep, out, flush=True)
    env.close()
print('DONE', flush=True)
