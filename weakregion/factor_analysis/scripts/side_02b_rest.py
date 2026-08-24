"""Side angle, step 2: probe sink position+orientation per layout via env instantiation.
One env per layout (style fixed at 11); dump sink pos, rot, robot base pos."""
import json
import sys
import numpy as np

LAYOUTS = [11, 13, 14, 16, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 39, 41, 42, 46, 47, 48, 49, 50, 53, 54, 55, 57, 59, 60]
OUT = '/data/xinyua11/tmp/factor_analysis_scratch/side_sinkpos_rest.json'

import robosuite
import robocasa  # noqa: F401  (registers kitchen envs)
from robosuite.controllers import load_composite_controller_config

res = {}
cc = load_composite_controller_config(robot='PandaOmron')
for L in LAYOUTS:
    try:
        env = robosuite.make(
            env_name='PickPlaceCounterToSink', robots='PandaOmron',
            controller_configs=cc, has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, use_object_obs=True, ignore_done=True, seed=0,
            layout_and_style_ids=[[L, 11]])
        env.reset()
        sink = env.get_fixture('sink')
        entry = {'sink_pos': [float(v) for v in np.asarray(sink.pos).ravel()]}
        for attr in ('rot', 'euler', 'quat'):
            if hasattr(sink, attr):
                v = getattr(sink, attr)
                try:
                    entry['sink_' + attr] = [float(x) for x in np.asarray(v).ravel()]
                except Exception:
                    entry['sink_' + attr] = str(v)
        try:
            entry['sink_size'] = [float(x) for x in np.asarray(sink.size).ravel()]
        except Exception:
            pass
        try:
            base = env.sim.data.get_body_xpos('base0_base')
            entry['robot_base'] = [float(x) for x in base]
        except Exception:
            try:
                entry['robot_base'] = [float(x) for x in env.robots[0].base_pos]
            except Exception:
                pass
        res[str(L)] = entry
        env.close()
        print('layout %d OK: %s' % (L, entry), flush=True)
    except Exception as e:
        print('layout %d FAIL: %r' % (L, e), flush=True)
        res[str(L)] = {'error': repr(e)}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
print('DONE', flush=True)
