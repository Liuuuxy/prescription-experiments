"""Probe robot base spawn position relative to sink for 5 layouts x 3 resets."""
import json
import numpy as np
import robosuite
import robocasa  # noqa: F401
from robosuite.controllers import load_composite_controller_config

OUT = '/data/xinyua11/tmp/factor_analysis_scratch/side_robotbase.json'
cc = load_composite_controller_config(robot='PandaOmron')
res = {}
for L in [56, 40, 15, 20, 33]:
    env = robosuite.make(env_name='PickPlaceCounterToSink', robots='PandaOmron',
                         controller_configs=cc, has_renderer=False, has_offscreen_renderer=False,
                         use_camera_obs=False, use_object_obs=True, ignore_done=True, seed=0,
                         layout_and_style_ids=[[L, 11]])
    entries = []
    for rep in range(3):
        env.reset()
        sink = env.get_fixture('sink')
        sp = np.asarray(sink.pos)[:2]
        yaw = sink.rot if np.isscalar(sink.rot) else sink.rot
        yaw = float(np.asarray(yaw).ravel()[0]) if not np.isscalar(yaw) else float(yaw)
        names = [nm for nm in env.sim.model.body_names if 'base' in nm.lower() or 'root' in nm.lower()]
        # robot root body
        root = env.robots[0].robot_model.root_body
        bp = np.array(env.sim.data.get_body_xpos(root))[:2]
        import numpy as _np
        R = _np.array([[_np.cos(-yaw), -_np.sin(-yaw)], [_np.sin(-yaw), _np.cos(-yaw)]])
        loc = R @ (bp - sp)
        entries.append(dict(root=root, base_world=[float(x) for x in bp],
                            base_sinkframe=[float(x) for x in loc]))
        if rep == 0:
            print('layout', L, 'bodies:', names[:8], flush=True)
    res[str(L)] = entries
    print('layout %d base sink-frame: %s' % (L, [np.round(e['base_sinkframe'], 2).tolist() for e in entries]), flush=True)
    env.close()
with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print('DONE', flush=True)
