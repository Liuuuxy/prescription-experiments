"""Quantitative analysis of policy rollouts on a RoboCasa task.

Pure functions over a list of per-episode records (plain dicts) so the analysis
logic can be unit-tested without running the simulator.

Each episode record is expected to contain at least::

    {
        "success": bool,
        "object_category": str,      # e.g. "peach"
        "obj_xy_rel": [dx, dy],      # object init pos relative to robot base
        "layout_id": int,
        "style_id": int,
    }

The headline output is an overall success rate with a confidence interval, plus
success-rate breakdowns by object category, spatial region, and scene. The
breakdowns are what turn "the model is bad" into "the model fails on <these
objects / in this region>" -- i.e. the weak-region signal that drives targeted
data collection.
"""

from __future__ import annotations

import math
from collections import defaultdict


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (better than normal
    approx at small n / extreme rates). Returns (low, high)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate_block(records):
    n = len(records)
    k = sum(1 for r in records if r["success"])
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "successes": k,
        "success_rate": (k / n) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _tertile_edges(values):
    """Two cut points splitting sorted values into ~equal thirds."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return (0.0, 0.0)
    return (s[n // 3], s[2 * n // 3])


def assign_regions(records, x_edges=None, y_edges=None):
    """Tag each record with a coarse spatial region label based on the object's
    init position relative to the robot base.

    By default, bin edges are tertiles of the observed positions (labels:
    left/center/right x near/mid/far). Pass explicit ``x_edges``/``y_edges``
    (each a 2-tuple) for cross-run comparability. Mutates records in place and
    also returns the edges used.
    """
    xs = [r["obj_xy_rel"][0] for r in records]
    ys = [r["obj_xy_rel"][1] for r in records]
    if x_edges is None:
        x_edges = _tertile_edges(xs)
    if y_edges is None:
        y_edges = _tertile_edges(ys)

    def _band(v, edges, labels):
        if v < edges[0]:
            return labels[0]
        if v < edges[1]:
            return labels[1]
        return labels[2]

    for r in records:
        # x ~ lateral (left/right), y ~ depth (near/far) in robot frame
        xlab = _band(r["obj_xy_rel"][0], x_edges, ("left", "center", "right"))
        ylab = _band(r["obj_xy_rel"][1], y_edges, ("near", "mid", "far"))
        r["region"] = f"{ylab}-{xlab}"
    return {"x_edges": list(x_edges), "y_edges": list(y_edges)}


def _group_rates(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    out = {str(g): _rate_block(rs) for g, rs in groups.items()}
    # sort by success_rate ascending so the weakest buckets surface first
    return dict(sorted(out.items(), key=lambda kv: kv[1]["success_rate"]))


def summarize(records, x_edges=None, y_edges=None):
    """Full quantitative summary: overall + breakdowns by object category,
    spatial region, layout, and style. Also flags the weakest buckets."""
    region_edges = assign_regions(records, x_edges, y_edges)
    summary = {
        "overall": _rate_block(records),
        "by_object_category": _group_rates(records, "object_category"),
        "by_region": _group_rates(records, "region"),
        "by_layout": _group_rates(records, "layout_id"),
        "by_style": _group_rates(records, "style_id"),
        "region_edges": region_edges,
    }
    summary["weakest"] = _weakest_buckets(summary)
    return summary


def _weakest_buckets(summary, min_n=3, top=3):
    """Across category/region breakdowns, surface the lowest success-rate
    buckets that have at least ``min_n`` samples -- the candidate targets for
    additional data."""
    cands = []
    for dim in ("by_object_category", "by_region"):
        for name, blk in summary[dim].items():
            if blk["n"] >= min_n:
                cands.append(
                    {
                        "dimension": dim,
                        "bucket": name,
                        "success_rate": blk["success_rate"],
                        "n": blk["n"],
                    }
                )
    cands.sort(key=lambda c: c["success_rate"])
    return cands[:top]


def format_report(summary, task=None, policy=None):
    """Human-readable text report."""
    lines = []
    h = "RoboCasa policy analysis"
    if task:
        h += f" -- task={task}"
    if policy:
        h += f" -- policy={policy}"
    lines.append(h)
    lines.append("=" * len(h))
    o = summary["overall"]
    lines.append(
        f"Overall: {o['successes']}/{o['n']} = {o['success_rate']:.1%} "
        f"(95% CI {o['ci95'][0]:.1%}-{o['ci95'][1]:.1%})"
    )

    def _section(title, block):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))
        for name, b in block.items():
            lines.append(
                f"  {name:<24} {b['success_rate']:>6.1%}  "
                f"({b['successes']}/{b['n']})"
            )

    _section("By object category (weakest first)", summary["by_object_category"])
    _section("By spatial region (weakest first)", summary["by_region"])
    _section("By layout", summary["by_layout"])
    _section("By style", summary["by_style"])

    lines.append("")
    lines.append("Weakest buckets (candidates for targeted data):")
    if summary["weakest"]:
        for c in summary["weakest"]:
            lines.append(
                f"  [{c['dimension'].replace('by_', '')}] {c['bucket']}: "
                f"{c['success_rate']:.1%} (n={c['n']})"
            )
    else:
        lines.append("  (not enough samples per bucket; increase --n)")
    return "\n".join(lines)
