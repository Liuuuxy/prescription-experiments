import json
from bandit_v1 import pool, config

def test_pool_table_complete_and_d0_flagged():
    df = pool.build_pool_table(write=False)
    assert len(df) == 9885
    assert df["episode_index"].is_unique
    d0 = set(json.load(open(config.ARMS_JSON))["base_episodes"])
    assert df["in_d0"].sum() == 400
    assert set(df.loc[df["in_d0"], "episode_index"]) == d0
    assert df["category"].nunique() >= 75
    assert pool.well_mask(df).sum() == 9485
