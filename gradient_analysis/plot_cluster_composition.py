"""Visualize gradient-cluster composition: CIFAR vs RoboCasa.
Top row: stacked composition bars (CIFAR: per-class segments; robot: region mix).
Bottom row: t-SNE of the whitened gradient sketches, colored by the label that
matters (CIFAR: rare/common class; robot: region)."""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

GA = "/data/xinyua11/robocasa/gradient_analysis"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
RARE_SH = ["#eb6834", "#d95926", "#f08a5c", "#c74d1e", "#f4a983", "#b34418"]
COMM_SH = ["#9fc0e8", "#7CA8DC", "#c3d7f0", "#5e94d1", "#d9e6f6", "#4a83c4"]
REGC = {"tall_vessel_grasp_fail": "#eb6834", "mid_band": "#2a78d6", "easy_band": "#1baf7a"}

def unit(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def whiten(X, k=10):
    U = unit(X.astype(np.float64)); Xc = U - U.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return unit(U - (U @ Vt[:k].T) @ Vt[:k])

# ---- CIFAR ----
G = "/data/xinyua11/xgradtest/gradlog"
meta = json.load(open(f"{G}/meta.json")); ti = meta["ckpts"].index(6000)
Xc = np.asarray(np.load(f"{G}/cand_raw.npy", mmap_mode="r")[ti], dtype=np.float64)
yc = np.load(f"{G}/pool_labels.npy"); n_rare = meta["n_rare"]; rare = yc < n_rare
names = pickle.load(open("/data/xinyua11/xgradtest/data/cifar-100-python/meta", "rb"))["fine_label_names"]
Wc = whiten(Xc)
lc = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(Wc)

# ---- Robot ----
eps, Ss = [], []
for d in (f"{GA}/sketches_pi0base_19999", f"{GA}/sketches_pi0base_19999_pool"):
    m = json.load(open(f"{d}/episodes.json")); eps += list(m["episodes"]); Ss.append(np.load(f"{d}/sketches.npy"))
S = np.vstack(Ss)
regmap = json.load(open(f"{GA}/demo_lists.json"))["region_assignment"]
keep = [i for i, e in enumerate(eps) if str(e) in regmap]
Sr = S[keep]; yreg = np.array([regmap[str(eps[i])] for i in keep])
Wr = whiten(Sr)
lr = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(Wr)

fig = plt.figure(figsize=(14, 10), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.32, wspace=0.18,
                      left=0.06, right=0.98, top=0.93, bottom=0.05)

# ---- (a) CIFAR composition bars ----
ax = fig.add_subplot(gs[0, 0])
order_c = np.argsort([-(rare[lc == c].mean()) for c in range(6)])
ylabels = []
for row, c in enumerate(order_c):
    m = lc == c
    cls, cnt = np.unique(yc[m], return_counts=True)
    o = np.argsort(-cnt)
    x = 0; ri = ci = 0
    for i in o:
        k, n = cls[i], cnt[i]
        if k < n_rare: col = RARE_SH[ri % len(RARE_SH)]; ri += 1
        else: col = COMM_SH[ci % len(COMM_SH)]; ci += 1
        frac = n / m.sum()
        ax.barh(row, frac, left=x, height=0.72, color=col, edgecolor="white", linewidth=0.4)
        if frac > 0.055:
            ax.text(x + frac / 2, row, names[k], ha="center", va="center", fontsize=7,
                    color=INK, rotation=0)
        x += frac
    ylabels.append(f"c{c}  n={m.sum()}\nrare {rare[m].mean():.0%}")
ax.set_yticks(range(6)); ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("fraction of cluster", fontsize=9)
ax.set_title("(a) CIFAR: each cluster's class composition — whole classes travel together\n"
             "orange shades = rare ('weak region') classes · blue shades = common classes",
             fontsize=10, loc="left")
ax.tick_params(labelsize=8); ax.invert_yaxis()
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ---- (b) Robot composition bars ----
ax = fig.add_subplot(gs[0, 1])
rows = [("pool", None)] + [(f"c{c}", c) for c in range(6)]
for row, (lab, c) in enumerate(rows):
    mix = yreg if c is None else yreg[lr == c]
    x = 0
    for r in ("tall_vessel_grasp_fail", "mid_band", "easy_band"):
        frac = (mix == r).mean()
        ax.barh(row, frac, left=x, height=0.72, color=REGC[r], edgecolor="white", linewidth=0.4,
                alpha=0.55 if c is None else 0.9)
        if row == 0:
            ax.text(x + frac / 2, row, r.split("_")[0], ha="center", va="center", fontsize=8, color=INK)
        x += frac
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] + ("" if r[1] is None else f"  n={(lr==r[1]).sum()}") for r in rows], fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("fraction of cluster", fontsize=9)
ax.set_title("(b) RoboCasa: every cluster's region mix = the pool's mix\n"
             "no cluster is enriched in anything — composition carries zero region info",
             fontsize=10, loc="left")
ax.tick_params(labelsize=8); ax.invert_yaxis()
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ---- t-SNE panels ----
rng = np.random.RandomState(0)
def tsne(W):
    idx = rng.choice(len(W), size=3000, replace=False)
    P = PCA(n_components=50, random_state=0).fit_transform(W[idx])
    E = TSNE(n_components=2, perplexity=40, random_state=0, init="pca").fit_transform(P)
    return idx, E

ax = fig.add_subplot(gs[1, 0])
idx, E = tsne(Wc)
r = rare[idx]
ax.scatter(E[~r, 0], E[~r, 1], s=4, c="#b9cbe0", linewidths=0, label="common classes")
ax.scatter(E[r, 0], E[r, 1], s=4, c="#eb6834", linewidths=0, label="rare (weak region)")
ax.set_title("(c) CIFAR gradients, t-SNE — rare classes form coherent clumps", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, markerscale=2.5); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color(GRID)

ax = fig.add_subplot(gs[1, 1])
idx, E = tsne(Wr)
yr = yreg[idx]
for rname, col in REGC.items():
    m = yr == rname
    ax.scatter(E[m, 0], E[m, 1], s=4, c=col, linewidths=0, label=rname.split("_")[0], alpha=0.8)
ax.set_title("(d) RoboCasa gradients, t-SNE — regions are salt-and-pepper everywhere", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, markerscale=2.5); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color(GRID)

fig.suptitle("Why gradient clusters mean something on CIFAR and nothing on the robot: same geometry, different content",
             fontsize=11.5, x=0.06, ha="left", color=INK)
out = f"{GA}/cluster_composition.png"
fig.savefig(out, dpi=150, facecolor="white")
print("wrote", out)
