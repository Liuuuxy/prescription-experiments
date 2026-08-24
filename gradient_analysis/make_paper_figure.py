"""Render the three-panel gradient figure at publication size (IEEEconf, two-column span).

Panels (all from real artifacts, no synthetic data):
  (a) training loss of three fine-tunes spanning the project's outcome range
  (b) per-demo gradient norms across each pull's own checkpoints, 6 content-arm pulls:
      own draw / never-trained pool demos / base-set (D0) demos
  (c) the same measurement for the execution-quality arms (style_hi vs style_lo)

Sources: bandit_v1/ledger/pull_logs/*_train_attempt1.log (loss),
         gradient_analysis/q2_absorption/<pull>/<step>_norms.npy (panels b,c),
         gradient_analysis/q3_pull_ablation.json (step-0 own-draw norms at pi_0).

Output: gradient_analysis/grad_three_panel.{pdf,png}  (7.0 x 2.35 in, 8pt type)
Run: /data/xinyua11/conda/envs/robocasa/bin/python gradient_analysis/make_paper_figure.py
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GA = "/data/xinyua11/robocasa/gradient_analysis"
LOGDIR = "/data/xinyua11/robocasa/bandit_v1/ledger/pull_logs"
STEPS = [5000, 10000, 15000, 19999]
CONTENT = ["tall_vessel_grasp_fail_j3", "tall_vessel_grasp_fail_j4",
           "gradarm_a_j3", "gradarm_a_j4", "gradarm_b_j3", "gradarm_b_j4"]
PI0_CTRL, PI0_D0 = 3.136, 0.554   # pi_0 reference levels (Q2 report)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE, ORANGE, GREEN, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#52514e"


def train_loss(pull, k=15):
    xs, ys = [], []
    for line in open(f"{LOGDIR}/{pull}_train_attempt1.log", errors="ignore"):
        m = re.match(r"Step (\d+):.*loss=([\d.]+)", line)
        if m:
            xs.append(int(m.group(1))); ys.append(float(m.group(2)))
    y = np.array(ys)
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k // 2, y[-1])])
    return np.array(xs), np.convolve(pad, np.ones(k) / k, mode="valid")


def absorb(pull):
    meta = json.load(open(f"{GA}/q2_absorption/{pull}/meta.json"))
    sets = np.array(meta["sets"])
    return {s: {t: float(np.load(f"{GA}/q2_absorption/{pull}/{s}_norms.npy")[sets == t].mean())
                for t in ("own", "ctrl", "d0")} for s in STEPS}


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
fig.subplots_adjust(left=0.068, right=0.995, top=0.86, bottom=0.20, wspace=0.30)
for ax in axes:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0")
    ax.set_axisbelow(True)

# (a) training loss
ax = axes[0]
for pull, col, lab in [("tall_vessel_grasp_fail_j4", BLUE, r"region arm ($+5.1$)"),
                       ("style_hi_j3", GREEN, r"quality-high ($+5.8$)"),
                       ("style_lo_j3", ORANGE, r"quality-low ($-3.1$)")]:
    x, y = train_loss(pull)
    ax.plot(x[::3], y[::3], color=col, label=lab)
ax.set_title(r"(a) training loss", loc="left")
ax.set_xlabel("fine-tune step"); ax.set_ylabel("flow-matching loss")
ax.set_xticks([0, 10000, 20000]); ax.set_xticklabels(["0", "10k", "20k"])
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25)

# (b) absorption / specialization, 6 content-arm pulls
ax = axes[1]
x0 = [0] + STEPS
per = {t: absorb(t) for t in CONTENT}
q3 = {r["pull"]: r for r in json.load(open(f"{GA}/q3_pull_ablation.json"))}
for tag, col, ls, lab, p0 in [("ctrl", GREY, (0, (3, 2)), "never trained on", PI0_CTRL),
                              ("d0", VIOLET, "-", "base set", PI0_D0),
                              ("own", BLUE, "-", "the pull's own 200", None)]:
    curves = [[(q3[t]["mean_gnorm"] if tag == "own" else p0)] + [per[t][s][tag] for s in STEPS]
              for t in CONTENT]
    for c in curves:
        ax.plot(x0, c, color=col, lw=0.5, alpha=0.30, ls=ls)
    ax.plot(x0, np.mean(curves, 0), color=col, ls=ls, label=lab)
ax.set_title(r"(b) demo gradients during training", loc="left")
ax.set_xlabel("checkpoint"); ax.set_ylabel(r"mean $\|\nabla_{\mathrm{LoRA}}\mathcal{L}\|$")
ax.set_xticks([0, 10000, 20000]); ax.set_xticklabels(["$\\pi_0$", "10k", "20k"])
ax.annotate("absorbed", xy=(5200, 0.45), xytext=(7000, 0.95), color=GREY, fontsize=6.8,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25,
          loc="upper left", bbox_to_anchor=(0.0, 1.02))

# (c) quality contrast
ax = axes[2]
hi, lo = absorb("style_hi_j3"), absorb("style_lo_j3")
ax.plot(x0, [PI0_CTRL] + [np.mean([hi[s]["ctrl"], lo[s]["ctrl"]]) for s in STEPS],
        color=GREY, ls=(0, (3, 2)), label="never trained on")
ax.plot(x0, [PI0_CTRL] + [hi[s]["own"] for s in STEPS], color=GREEN, label=r"quality-high own")
ax.plot(x0, [PI0_CTRL] + [lo[s]["own"] for s in STEPS], color=ORANGE, label=r"quality-low own")
ax.set_title(r"(c) quality: the one visible difference", loc="left")
ax.set_xlabel("checkpoint"); ax.set_ylabel(r"mean $\|\nabla_{\mathrm{LoRA}}\mathcal{L}\|$")
ax.set_xticks([0, 10000, 20000]); ax.set_xticklabels(["$\\pi_0$", "10k", "20k"])
ax.annotate("+46%", xy=(19999, 0.70), xytext=(15500, 1.35), color=GREY, fontsize=6.8,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.legend(frameon=False, handlelength=1.4, borderpad=0.2, labelspacing=0.25)

for ext in ("pdf", "png"):
    fig.savefig(f"{GA}/grad_three_panel.{ext}", dpi=400 if ext == "png" else None,
                bbox_inches="tight", pad_inches=0.02)
print(f"wrote {GA}/grad_three_panel.pdf and .png")
