"""Per-style and per-layout success-rate plots (id vs SR), checkpoint-controlled.

Run-adjusted SR = within-run demeaned success + grand mean. 95% CI via within-run
cluster bootstrap. Grey band = 2.5-97.5 pct envelope of per-id adjusted SR under
within-run label permutation (the noise floor: ids inside it are indistinguishable
from chance). Bars colored diverging around the grand mean (red=below avg/harder,
blue=above avg/easier).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

rng = np.random.default_rng(0)
FA = "/data/xinyua11/robocasa/weakregion/factor_analysis"
df = pd.read_csv(FA + "/pooled_episodes.csv")
df["success"] = df["success"].astype(int)
runs = df["run"].values
gm = df["success"].mean()
run_mean = df.groupby("run")["success"].transform("mean").values
df["sr_dm"] = df["success"].values - run_mean

def adjusted_sr(sub):
    return sub["sr_dm"].mean() + gm

def per_id_stats(col, B=2000):
    ids = np.sort(df[col].unique())
    rows = []
    # precompute run-block indices for bootstrap
    run_groups = {r: np.where(runs == r)[0] for r in np.unique(runs)}
    for i in ids:
        m = df[col].values == i
        n = int(m.sum())
        adj = df.loc[m, "sr_dm"].mean() + gm
        raw = df.loc[m, "success"].mean()
        rows.append(dict(id=int(i), n=n, sr_adj=adj, sr_raw=raw))
    stat = pd.DataFrame(rows)
    # within-run cluster bootstrap CI on adjusted SR
    idx_by_run = [run_groups[r] for r in np.unique(runs)]
    dm = df["sr_dm"].values
    idv = df[col].values
    boot = {int(i): [] for i in ids}
    for _ in range(B):
        samp = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in idx_by_run])
        d_s, id_s = dm[samp], idv[samp]
        for i in ids:
            mm = id_s == i
            if mm.any():
                boot[int(i)].append(d_s[mm].mean() + gm)
    lo = np.array([np.percentile(boot[int(i)], 2.5) for i in ids])
    hi = np.array([np.percentile(boot[int(i)], 97.5) for i in ids])
    stat["ci_lo"], stat["ci_hi"] = lo, hi
    # permutation null envelope (shuffle id labels within run)
    null_spreads = np.zeros((500, len(ids)))
    for b in range(500):
        perm_id = np.empty_like(idv)
        for g in idx_by_run:
            perm_id[g] = rng.permutation(idv[g])
        d = df["sr_dm"].values
        for k, i in enumerate(ids):
            mm = perm_id == i
            null_spreads[b, k] = d[mm].mean() + gm if mm.any() else gm
    null_lo, null_hi = np.percentile(null_spreads, 2.5), np.percentile(null_spreads, 97.5)
    return stat, (null_lo, null_hi)

def draw(col, title, hardest, easiest, fname):
    stat, (nlo, nhi) = per_id_stats(col)
    norm = TwoSlopeNorm(vmin=stat.sr_adj.min(), vcenter=gm, vmax=stat.sr_adj.max())
    cmap = cm.get_cmap("RdBu")
    def colors(vals): return [cmap(norm(v)) for v in vals]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    # -- panel A: id order --
    ax = axes[0]
    s = stat.sort_values("id")
    x = np.arange(len(s))
    ax.axhspan(nlo, nhi, color="0.85", zorder=0, label="noise band (perm. null 95%)")
    ax.axhline(gm, color="0.35", lw=1, ls="--", zorder=1, label=f"grand mean {gm:.3f}")
    ax.bar(x, s.sr_adj, color=colors(s.sr_adj), zorder=2, width=0.8)
    ax.errorbar(x, s.sr_adj, yerr=[s.sr_adj - s.ci_lo, s.ci_hi - s.sr_adj],
                fmt="none", ecolor="0.3", elinewidth=0.7, capsize=0, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(s.id, rotation=90, fontsize=6)
    ax.set_xlabel(f"{col} (in id order)"); ax.set_ylabel("run-adjusted success rate")
    ax.set_title(f"{title} — by id"); ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0.01)
    # -- panel B: sorted --
    ax = axes[1]
    s = stat.sort_values("sr_adj").reset_index(drop=True)
    x = np.arange(len(s))
    ax.axhspan(nlo, nhi, color="0.85", zorder=0)
    ax.axhline(gm, color="0.35", lw=1, ls="--", zorder=1)
    ax.bar(x, s.sr_adj, color=colors(s.sr_adj), zorder=2, width=0.85)
    ax.errorbar(x, s.sr_adj, yerr=[s.sr_adj - s.ci_lo, s.ci_hi - s.sr_adj],
                fmt="none", ecolor="0.3", elinewidth=0.7, capsize=0, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(s.id, rotation=90, fontsize=6)
    ax.set_xlabel(f"{col} (sorted worst -> best)"); ax.set_ylabel("run-adjusted success rate")
    ax.set_title(f"{title} — sorted (spread {stat.sr_adj.max()-stat.sr_adj.min():.2f})")
    plt.tight_layout()
    plt.savefig(FA + "/plots/" + fname, dpi=130, bbox_inches="tight")
    print("wrote", fname, "| adj-SR range %.3f-%.3f" % (stat.sr_adj.min(), stat.sr_adj.max()),
          "| null band %.3f-%.3f" % (nlo, nhi))
    stat.to_csv(FA + f"/perid_{col}_stats.csv", index=False)
    return stat

draw("style_id", "Success rate by kitchen STYLE", [49,43,42,21,22], [27,36,31,30,17], "sr_by_style.png")
draw("layout_id", "Success rate by kitchen LAYOUT", [], [], "sr_by_layout.png")
