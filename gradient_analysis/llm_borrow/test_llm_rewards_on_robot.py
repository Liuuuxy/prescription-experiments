"""Do the LLM data-mixing bandits' REWARD SIGNALS rank arms on OUR stack?

Zero-GPU, read-only. Tests three reward families from the LLM bandit literature
against the bandit_v1 ledger's realized closed-loop deltas:

  ODM (Albalak 2023)          reward = training loss on the arm's data
  Graves 2017 TPG / Skill-it  reward = loss DROP on a target validation set
  Graves 2017 GPG             reward = ||grad L||^2 of the arm's data at the base ckpt

Sources: gradient_analysis/loss_probe_calibration.parquet (per-pull flow-matching
loss on target / balanced / retention demo sets, incl. a pi0base_ref row),
gradient_analysis/q3_pull_ablation.json (per-draw pi0-time gradient stats),
bandit_v1 ledger pulls.parquet (realized delta = closed-loop success - baseline).

Writes llm_borrow/reward_signal_test.json.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/data/xinyua11/robocasa")
from bandit_v1 import ledger  # noqa: E402

GA = "/data/xinyua11/robocasa/gradient_analysis"
OUT = f"{GA}/llm_borrow/reward_signal_test.json"


def corr(x, y):
    r, rp = stats.pearsonr(x, y)
    s, sp = stats.spearmanr(x, y)
    return dict(pearson_r=float(r), pearson_p=float(rp),
                spearman_rho=float(s), spearman_p=float(sp), n=int(len(x)))


def main():
    pulls = ledger.read("pulls")
    pulls = pulls[pulls.status == "ok"]
    lp = pd.read_parquet(f"{GA}/loss_probe_calibration.parquet")
    ref = lp[lp.pull_id == "pi0base_ref"].iloc[0]
    m = lp.merge(pulls[["pull_id", "arm", "round_j", "delta"]], on="pull_id")
    m["tpg"] = ref.loss_target - m.loss_target        # Graves target prediction gain
    m["bpg"] = ref.loss_balanced - m.loss_balanced
    nn = m[m.arm != "null"].copy()                    # all added B=200: composition-only

    res = {"n_pulls_with_loss_probe": int(len(m)), "n_non_null": int(len(nn))}
    res["loss_reward_all_pulls"] = {c: corr(m[c], m.delta)
                                    for c in ["loss_target", "tpg", "loss_balanced", "bpg"]}
    res["loss_reward_non_null"] = {c: corr(nn[c], nn.delta)
                                   for c in ["loss_target", "tpg", "loss_balanced",
                                             "loss_retention"]}
    res["within_round_kendall_neg_loss_target_vs_delta"] = {}
    for rj, g in nn.groupby("round_j"):
        if len(g) < 3:
            continue
        t, tp = stats.kendalltau(-g.loss_target, g.delta)
        res["within_round_kendall_neg_loss_target_vs_delta"][int(rj)] = dict(
            n=int(len(g)), tau=float(t), p=float(tp), arms=list(g.arm))

    # Which arm would an ODM/Skill-it style loss reward have funded?
    best = nn.loc[nn.groupby("round_j").loss_target.idxmin()]
    res["arm_the_loss_reward_would_pick_per_round"] = {
        int(r.round_j): dict(arm=r.arm, loss_target=float(r.loss_target),
                             realized_delta_pp=float(r.delta * 100)) for _, r in best.iterrows()}
    res["arm_mean_delta_pp"] = {a: float(g.delta.mean() * 100)
                                for a, g in nn.groupby("arm")}

    q3 = pd.DataFrame(json.load(open(f"{GA}/q3_pull_ablation.json")))
    res["graves_gpg_and_batch_stats"] = {c: corr(q3[c], q3.delta) for c in
                                         ["mean_gnorm", "coherence", "wsep",
                                          "self_sim", "cos_region"]}
    res["noise_floor_pp"] = 3.3
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
