"""Measure c DIRECTLY on real GPU: the correlation between the CHEAP SIGNAL and the
TRUE arm value, for each corruption of the reward channel.

The transfer map (f4_transfer.py synth) is indexed by c = corr(cheap signal, true arm
value). This script measures c for the REAL CIFAR system under each F4 corruption, so
the map can be read at the measured point instead of a hypothetical one. It is ~100x
cheaper than racing allocators, because it measures the reward channel itself:

  for each arm, from the SAME base checkpoint, run one burst and read the loss drop on
  several probe slices at once --
     clean      : held-out TARGET-class examples          (aligned reward channel)
     offtarget  : held-out WELL-LEARNED-class examples    (F4 mechanism B)
     noisy_rho  : the clean slice with a fraction rho of labels randomized (mechanism A)
  then correlate each arm-ranking with the armrace ground truth (delta rare-class acc).

Also scores the JEST channel (learner loss - reference loss on each arm's own data)
under clean and randomized labels.

  CUDA_VISIBLE_DEVICES=1 python f4_signal_probe.py --seeds 0,1,2 --steps 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f4_transfer import spearman  # noqa: E402
from wind_tunnel import HERE, B_DRAW, CifarBackend  # noqa: E402


def build_probes(bk, rhos, n=2048, seed=1234):
    torch, np_ = bk.torch, np
    y = np_.array(bk.tr.targets)
    rest = np_.array(bk.sp["pool"])[len(bk.pool):]      # unreachable by any arm
    on, off = rest[y[rest] < bk.N_RARE], rest[y[rest] >= bk.N_RARE]
    rng = np_.random.default_rng(seed)
    probes = {"clean": on[:n], "offtarget": off[:n]}
    labs = {"clean": None, "offtarget": None}
    for r in rhos:
        lab = y.copy()
        m = rng.random(len(lab)) < r
        lab[m] = rng.integers(0, int(lab.max()) + 1, size=int(m.sum()))
        probes[f"noisy{r}"] = on[:n]
        labs[f"noisy{r}"] = torch.as_tensor(lab, device=bk.DEV).long()
    idx = {k: torch.as_tensor(v, device=bk.DEV).long() for k, v in probes.items()}
    return idx, labs


def probe_loss(bk, model, ids, lab):
    torch = bk.torch
    model.eval()
    tot = 0.0
    Y = bk.Ytr if lab is None else lab
    with torch.no_grad():
        for s in range(0, len(ids), 512):
            b = ids[s:s + 512]
            tot += torch.nn.functional.cross_entropy(
                model(bk._norm(bk.Xtr[b])), Y[b], reduction="sum").item()
    return tot / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--rhos", default="0.25,0.5,1.0")
    ap.add_argument("--out", default=f"{HERE}/f4_signal_probe.json")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    rhos = [float(x) for x in a.rhos.split(",")]

    bk = CifarBackend(cache_path=f"{HERE}/cache_probe.json", verbose=False)
    bk._setup()
    truth = bk.true_values()
    arms = list(bk.arms)
    idx, labs = build_probes(bk, rhos)
    t0 = time.time()

    # ---- channel 1: loss progress on each probe slice
    prog = {k: {a_: [] for a_ in arms} for k in idx}
    for sd in seeds:
        for arm in arms:
            st = bk.session_init(sd)
            before = {k: probe_loss(bk, st["model"], idx[k], labs[k]) for k in idx}
            bk.session_burst(st, arm, a.steps)
            after = {k: probe_loss(bk, st["model"], idx[k], labs[k]) for k in idx}
            for k in idx:
                prog[k][arm].append((before[k] - after[k]) / a.steps * 1000.0)
            print(f"[probe] seed {sd} arm {arm:10s} " + " ".join(
                f"{k}={(before[k]-after[k])/a.steps*1000:+.4f}" for k in idx), flush=True)

    tv = [truth[x] for x in arms]
    out = {"steps": a.steps, "seeds": seeds, "truth_pp": {k: round(v * 100, 3) for k, v in truth.items()},
           "loss_progress_channel": {}, "elapsed_s": round(time.time() - t0, 1)}
    for k in idx:
        mean = [float(np.mean(prog[k][x])) for x in arms]
        per_seed = [[prog[k][x][i] for x in arms] for i in range(len(seeds))]
        out["loss_progress_channel"][k] = {
            "mean_progress": dict(zip(arms, [round(m, 5) for m in mean])),
            "pearson_vs_truth": round(float(np.corrcoef(mean, tv)[0, 1]), 3),
            "spearman_vs_truth": round(spearman(mean, tv), 3),
            "argmax_arm": arms[int(np.argmax(mean))],
            "true_argmax": max(truth, key=truth.get),
            "per_seed_spearman": [round(spearman(p, tv), 3) for p in per_seed],
            "per_seed_argmax": [arms[int(np.argmax(p))] for p in per_seed]}
        d = out["loss_progress_channel"][k]
        print(f"[probe] CHANNEL {k:12s} spearman {d['spearman_vs_truth']:+.3f} "
              f"pearson {d['pearson_vs_truth']:+.3f} argmax {d['argmax_arm']} "
              f"(truth {d['true_argmax']}) per-seed spearman {d['per_seed_spearman']}",
              flush=True)

    # ---- channel 2: JEST learnability on each arm's own data, clean vs noisy labels
    jest = {}
    for sd in seeds[:2]:
        ref = bk.make_reference(a.steps, sd)
        learner = bk._fresh_model(0)
        for kind, lab in [("clean", None)] + [(f"noisy{r}", labs[f"noisy{r}"]) for r in rhos]:
            sc = {}
            for arm in arms:
                if not bk.arm_is_scorable(arm):
                    continue
                ids = bk._draw(arm, 9999, min(512, len(bk.arm_positions[arm])))
                ids_t = bk.torch.as_tensor(ids, device=bk.DEV).long()
                sc[arm] = (probe_loss(bk, learner, ids_t, lab)
                           - probe_loss(bk, ref["model"], ids_t, lab))
            jest.setdefault(kind, []).append(sc)
    out["jest_channel"] = {}
    for kind, runs in jest.items():
        keys = list(runs[0])
        mean = [float(np.mean([r[x] for r in runs])) for x in keys]
        tvj = [truth[x] for x in keys]
        out["jest_channel"][kind] = {
            "mean_score": dict(zip(keys, [round(m, 4) for m in mean])),
            "spearman_vs_truth": round(spearman(mean, tvj), 3),
            "pearson_vs_truth": round(float(np.corrcoef(mean, tvj)[0, 1]), 3),
            "argmax_arm": keys[int(np.argmax(mean))], "n_seeds": len(runs),
            "note": "null arm is unscorable by JEST (no data)"}
        print(f"[probe] JEST {kind:10s} spearman "
              f"{out['jest_channel'][kind]['spearman_vs_truth']:+.3f} argmax "
              f"{out['jest_channel'][kind]['argmax_arm']}", flush=True)

    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"[probe] wrote {a.out} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
