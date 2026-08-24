"""Which EXPLICIT criterion organizes the robot gradient space?
Tests, on the same whitened sketches/k-means as the cluster analysis:
  labels: object category (79), side of sink (L/R), height terciles, |x_rel|
  (distance) terciles, trajectory-length terciles, region (tall/mid/easy, ref)
  metrics per label: NMI with the 6 gradient clusters; and DIRECT encoding via
  split-half contrast AUC (binary/extreme-tercile versions, 300 vs 300, held out)
  plus max |corr| of the top-20 whitened PCs with each continuous knob."""
import json, sys
import numpy as np
sys.path.insert(0, "/data/xinyua11/robocasa")
sys.path.insert(0, "/data/xinyua11/robocasa/gradient_analysis")
from analyze_gate import split_half_contrast_auc, unitize
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")
from bandit_v1 import ledger

GA = "/data/xinyua11/robocasa/gradient_analysis"
eps, Ss = [], []
for d in (f"{GA}/sketches_pi0base_19999", f"{GA}/sketches_pi0base_19999_pool"):
    m = json.load(open(f"{d}/episodes.json")); eps += list(m["episodes"]); Ss.append(np.load(f"{d}/sketches.npy"))
S = np.vstack(Ss)
regmap = json.load(open(f"{GA}/demo_lists.json"))["region_assignment"]
pdf = ledger.read("pool_demos").set_index("episode_index")
MG = "/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot"
L = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(f"{MG}/meta/episodes.jsonl")}

keep = [i for i, e in enumerate(eps) if str(e) in regmap and e in pdf.index and e in L]
er = [eps[i] for i in keep]
W = S[keep]
U = unitize(W.astype(np.float64)); Xc = U - U.mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
Wh = unitize(U - (U @ Vt[:10].T) @ Vt[:10])
km = KMeans(n_clusters=6, n_init=10, random_state=0).fit_predict(Wh)

sub = pdf.loc[er]
h = sub.h.to_numpy(); xr = sub.x_rel.abs().to_numpy(); side = (sub.side.to_numpy() > 0)
cat = sub.category.to_numpy(); ln = np.array([L[e] for e in er]); reg = np.array([regmap[str(e)] for e in er])
def terc(v):
    q = np.quantile(v, [1/3, 2/3]); return np.digitize(v, q)

rng = np.random.RandomState(7)
def gate(mask_pos, mask_neg, n=300):
    ip = np.where(mask_pos)[0]; im = np.where(mask_neg)[0]
    a = rng.choice(ip, min(n, len(ip)), replace=False); b = rng.choice(im, min(n, len(im)), replace=False)
    return split_half_contrast_auc(Wh[a], Wh[b])

print(f"n={len(er)} demos | metric 1: NMI(label, 6 gradient clusters) | metric 2: held-out direction AUC")
print(f"{'criterion':28s} {'NMI':>7s} {'AUC':>7s}")
rows = [
    ("region tall/mid/easy (ref)", NMI(reg, km), gate(reg == 'tall_vessel_grasp_fail', reg == 'easy_band')),
    ("object category (79)",       NMI(cat, km), np.nan),
    ("side of sink (L vs R)",      NMI(side, km), gate(side, ~side)),
    ("height (top vs bottom terc)", NMI(terc(h), km), gate(terc(h) == 2, terc(h) == 0)),
    ("|x_rel| dist (top vs bot)",  NMI(terc(xr), km), gate(terc(xr) == 2, terc(xr) == 0)),
    ("traj length (top vs bot)",   NMI(terc(ln), km), gate(terc(ln) == 2, terc(ln) == 0)),
]
for name, nmi, auc in rows:
    a = f"{auc:7.3f}" if auc == auc else "      -"
    print(f"{name:28s} {nmi:7.3f} {a}")

P = PCA(n_components=20, random_state=0).fit_transform(Wh)
print("\nmax |corr| of any top-20 whitened PC with each continuous knob:")
for name, v in (("height", h), ("|x_rel|", xr), ("traj length", ln), ("side(+/-)", sub.side.to_numpy())):
    c = max(abs(float(np.corrcoef(P[:, i], v)[0, 1])) for i in range(20))
    print(f"  {name:12s} {c:.3f}")
