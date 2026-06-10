"""Cross-model triage over RoboCasa eval outputs (project step 5.4).

Goal: "don't waste time on very bad or very good models." Ingests the eval
outputs produced by the different benchmark harnesses (which use *different*
formats) and emits one normalized comparison: success rate + Wilson CI + a
triage verdict per model, plus a pointer to success/failure videos for quick
qualitative spot-checks.

Formats handled:
  * Diffusion Policy  -> eval_log.json  ("success_rate/<task>", per-episode
                         "test/sim_max_reward_*", "test/sim_video_*")
  * pi0 (openpi)      -> stats.json     ("num_episodes", "success_rate") with
                         sibling rollout_<i>_{success,failure}.mp4 videos
  * GR00T             -> (added once its eval output format is known)

Usage:
  python compare_models.py            # auto-discovers under ../evals
  python compare_models.py --evals-root <dir> --out report_models.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from analysis import wilson_ci


def _verdict(rate, n):
    """Triage bucket. Wider language at small n (low confidence)."""
    lo, hi = wilson_ci(round(rate * n), n)
    if n < 10:
        conf = " (LOW confidence, n<10)"
    else:
        conf = ""
    if rate < 0.10:
        v = "very bad — skip / needs major work"
    elif rate < 0.40:
        v = "weak"
    elif rate < 0.70:
        v = "partial"
    elif rate < 0.90:
        v = "good"
    else:
        v = "very good — near-solved"
    return v + conf


def _parse_dp(path):
    with open(path) as f:
        d = json.load(f)
    task = next((k.split("/", 1)[1] for k in d if k.startswith("success_rate/")), "?")
    rate = next((v for k, v in d.items() if k.startswith("success_rate/")), None)
    n = sum(1 for k in d if k.startswith("test/sim_max_reward_"))
    videos = [v for k, v in d.items() if k.startswith("test/sim_video_")]
    return {"task": task, "rate": rate, "n": n, "videos": videos,
            "video_dir": os.path.dirname(path)}


def _parse_pi0(path):
    with open(path) as f:
        d = json.load(f)
    n = d.get("num_episodes")
    rate = d.get("success_rate")
    vids = sorted(glob.glob(os.path.join(os.path.dirname(path), "rollout_*.mp4")))
    # task name is the parent dir two levels up (.../<task>/<timestamp>/stats.json)
    task = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return {"task": task, "rate": rate, "n": n, "videos": vids,
            "video_dir": os.path.dirname(path)}


def discover(evals_root):
    """Return list of (model_label, kind, path)."""
    found = []
    for p in glob.glob(os.path.join(evals_root, "**", "eval_log.json"), recursive=True):
        label = p[len(evals_root):].strip("/").split("/")[0]
        found.append((label, "dp", p))
    for p in glob.glob(os.path.join(evals_root, "**", "stats.json"), recursive=True):
        label = p[len(evals_root):].strip("/").split("/")[0]
        found.append((label, "pi0", p))
    return sorted(found)


def build_rows(evals_root):
    rows = []
    for label, kind, path in discover(evals_root):
        rec = _parse_dp(path) if kind == "dp" else _parse_pi0(path)
        if rec["rate"] is None or not rec["n"]:
            continue
        k = round(rec["rate"] * rec["n"])
        lo, hi = wilson_ci(k, rec["n"])
        n_succ = sum(1 for v in rec["videos"] if "success" in os.path.basename(v))
        n_fail = sum(1 for v in rec["videos"] if "failure" in os.path.basename(v))
        rows.append({
            "model": label, "task": rec["task"], "rate": rec["rate"], "n": rec["n"],
            "k": k, "ci": (lo, hi), "verdict": _verdict(rec["rate"], rec["n"]),
            "n_videos": len(rec["videos"]), "n_success_vid": n_succ,
            "n_fail_vid": n_fail, "video_dir": rec["video_dir"],
        })
    rows.sort(key=lambda r: r["rate"])  # weakest first
    return rows


def format_report(rows):
    lines = ["# Model triage (quant + qual)\n",
             "Success rate with Wilson 95% CI, weakest first. "
             "Use the verdict to decide whether a model is worth deeper analysis; "
             "open the video_dir to spot-check *how* it succeeds/fails.\n",
             "| model | task | success | 95% CI | n | verdict | videos |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['task']} | {r['rate']:.0%} ({r['k']}/{r['n']}) | "
            f"{r['ci'][0]:.0%}–{r['ci'][1]:.0%} | {r['n']} | {r['verdict']} | "
            f"{r['n_videos']} ({r['n_success_vid']}✓/{r['n_fail_vid']}✗) |"
        )
    lines.append("\nVideo dirs:")
    for r in rows:
        lines.append(f"- **{r['model']}**: `{r['video_dir']}`")
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--evals-root",
                   default=os.path.join(here, "..", "evals"))
    p.add_argument("--out", default=os.path.join(here, "..", "model_triage.md"))
    args = p.parse_args()
    rows = build_rows(os.path.abspath(args.evals_root))
    report = format_report(rows)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote {args.out}")
