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


def test_row_level_fidelity():
    """Spot-check the table against fx_pool.json's raw rows at i=0, 1824, 9884,
    and lock down the canonical int side encoding decided in task 2."""
    fx = json.load(open(config.FX_POOL_JSON))
    fields = fx["fields"]
    cats = fx["cats"]
    idx = {f: i for i, f in enumerate(fields)}
    rows = fx["rows"]

    df = pool.build_pool_table(write=False)
    by_episode = df.set_index("episode_index")

    for i in (0, 1824, 9884):
        src = rows[i]
        expected_category = cats[int(src[idx["cat"]])]["name"]
        expected_side = int(src[idx["side"]])
        table_row = by_episode.loc[i]
        assert table_row["category"] == expected_category
        assert table_row["side"] == expected_side

    assert df["side"].dtype.kind == "i"
    assert set(df["side"].unique()) <= {-1, 1}
