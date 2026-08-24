"""Tables + the transfer-condition map. Reads the part1_*/f4_* JSONs, writes nothing
except an optional summary JSON; every claim is printed with its uncertainty.

  python analyze.py part1 part1_synthetic_robot.json
  python analyze.py f4    f4_synth_robot.json --snr 1
  python analyze.py map   f4_synth_robot.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_lib import b95, paired_vs, wilson  # noqa: E402

ORDER = ["oracle", "successive_halving", "jest_then_confirm", "jest_then_confirm_k3",
         "jest_then_confirm_k4", "jest", "learning_progress_exp3",
         "learning_progress_ucb", "regmix", "random"]


def _sortkey(n):
    return ORDER.index(n) if n in ORDER else 99


def load(p):
    """Load one JSON, or merge several comma-separated ones (later files win)."""
    parts = [json.load(open(x)) for x in p.split(",")]
    out = parts[0]
    for q in parts[1:]:
        out["cells"].update(q["cells"])
    return out


def part1(args):
    d = load(args.path)
    cells = d["cells"]
    truth = d["truth"]
    best = max(truth, key=truth.get)
    budgets = sorted({float(k.split("|")[0]) for k in cells})
    allocs = sorted({k.split("|")[1] for k in cells}, key=_sortkey)
    print(f"\n=== {os.path.basename(args.path)}  reps={d['reps']}  "
          f"best arm = {best} ({truth[best]*100:+.2f}pp) ===")
    print("true values (pp): " + "  ".join(f"{a}={v*100:+.2f}" for a, v in
                                           sorted(truth.items(), key=lambda kv: -kv[1])))
    for metric, fmt, scale in (("mean_regret", "{:>13s}", 100.0),
                               ("p_correct", "{:>13s}", 1.0)):
        print(f"\n{'SIMPLE REGRET (pp), mean +/- se' if metric=='mean_regret' else 'P(picks the truly best arm) [95% Wilson]'}")
        print(f"{'allocator':26s}" + "".join(f"{b:>15.1f}" for b in budgets))
        for a in allocs:
            row = f"{a:26s}"
            for b in budgets:
                c = cells.get(f"{b}|{a}")
                if not c:
                    row += f"{'-':>15s}"
                elif metric == "mean_regret":
                    row += f"{c['mean_regret']*100:9.3f}+-{c['se_regret']*100:4.3f}"
                else:
                    lo, hi = c["p_correct_ci"]
                    row += f"{c['p_correct']:7.2f}[{lo:.2f},{hi:.2f}]"
            print(row)
    print(f"\nBUDGET-TO-95%-OF-ORACLE (FTE; '>{budgets[-1]:.0f}' = never on this grid)")
    for a in allocs:
        by_b = {b: cells[f"{b}|{a}"] for b in budgets if f"{b}|{a}" in cells}
        v = b95(by_b)
        print(f"  {a:26s} {'%.2f' % v if v is not None else '>%.0f' % budgets[-1]:>8s}"
              f"    (value/oracle at max budget: "
              f"{by_b[max(by_b)]['mean_value']/by_b[max(by_b)]['best_value']:.3f})")
    print(f"\nSPEND (mean FTE spent of budget) and unspent")
    print(f"{'allocator':26s}" + "".join(f"{b:>12.1f}" for b in budgets))
    for a in allocs:
        row = f"{a:26s}"
        for b in budgets:
            c = cells.get(f"{b}|{a}")
            row += f"{c['mean_spent_fte']:12.2f}" if c else f"{'-':>12s}"
        print(row)
    # paired vs random
    print(f"\nPAIRED vs the random floor (regret difference, negative = better; "
          f"bootstrap 95% CI)")
    for b in budgets:
        r = cells.get(f"{b}|random")
        if not r:
            continue
        print(f"  budget {b:g} FTE")
        for a in allocs:
            if a in ("random",) or f"{b}|{a}" not in cells:
                continue
            pv = paired_vs(cells[f"{b}|{a}"], r)
            tag = "BEATS" if pv["beats_ref"] else ("LOSES" if pv["loses_to_ref"] else "tied")
            print(f"    {a:26s} {pv['delta_regret']*100:+7.3f}pp "
                  f"[{pv['ci95'][0]*100:+.3f},{pv['ci95'][1]*100:+.3f}]  {tag}")


def f4_table(args):
    d = load(args.path)
    cells = d["cells"]
    snrs = sorted({c["snr"] for c in cells.values()})
    budgets = sorted({c["budget_fte"] for c in cells.values()})
    cs = sorted({c["c"] for c in cells.values()}, reverse=True)
    allocs = sorted({c["allocator"] for c in cells.values()}, key=_sortkey)
    for b in budgets:
        for snr in snrs:
            if args.snr is not None and abs(snr - args.snr) > 1e-9:
                continue
            print(f"\n=== budget {b:g} FTE | signal SNR {snr:g} | "
                  f"reps {d['reps']} ({d.get('n_orth_directions','?')} nuisance dirs) ===")
            print("MEAN REGRET (pp).  c = correlation(cheap signal, true arm value)")
            print(f"{'allocator':26s}" + "".join(f"{c:>8.2f}" for c in cs))
            for a in allocs:
                row = f"{a:26s}"
                for c in cs:
                    k = f"{b}|{snr}|{c}|{a}"
                    row += f"{cells[k]['mean_regret']*100:8.2f}" if k in cells else f"{'-':>8s}"
                print(row)
            print("\nvs RANDOM FLOOR (paired):  + = beats random, ~ = within noise, "
                  "X = WORSE than random")
            print(f"{'allocator':26s}" + "".join(f"{c:>8.2f}" for c in cs))
            for a in allocs:
                if a in ("random", "oracle"):
                    continue
                row = f"{a:26s}"
                for c in cs:
                    k, kr = f"{b}|{snr}|{c}|{a}", f"{b}|{snr}|{c}|random"
                    if k not in cells or kr not in cells:
                        row += f"{'-':>8s}"
                        continue
                    pv = paired_vs(cells[k], cells[kr], n_boot=4000)
                    row += f"{'+' if pv['beats_ref'] else ('X' if pv['loses_to_ref'] else '~'):>8s}"
                print(row)


def transfer_map(args):
    """c* per allocator: the correlation at which it stops BEATING the random floor
    (highest c whose paired CI no longer clears 0), reported as a grid interval."""
    d = load(args.path)
    cells = d["cells"]
    budgets = sorted({c["budget_fte"] for c in cells.values()})
    snrs = sorted({c["snr"] for c in cells.values()}, reverse=True)
    cs = sorted({c["c"] for c in cells.values()}, reverse=True)
    allocs = [a for a in sorted({c["allocator"] for c in cells.values()}, key=_sortkey)
              if a not in ("random", "oracle")]
    for ref_name, label in (("random", "the RANDOM FLOOR"),
                            ("successive_halving", "our incumbent SUCCESSIVE HALVING")):
        print(f"\n=== TRANSFER-CONDITION MAP vs {label} ===")
        print("c* = the LOWEST signal/outcome correlation at which the method still")
        print(f"     beats {ref_name} (paired bootstrap CI95 < 0). 'never' = it never")
        print("     beats it, even at c=1 (perfect transfer).")
        for b in budgets:
            print(f"\nbudget {b:g} FTE")
            print(f"{'allocator':26s}" + "".join(f"{('SNR %g' % s):>12s}" for s in snrs))
            for a in allocs:
                if a == ref_name:
                    continue
                row = f"{a:26s}"
                for s in snrs:
                    good = []
                    for c in cs:
                        k, kr = f"{b}|{s}|{c}|{a}", f"{b}|{s}|{c}|{ref_name}"
                        if k in cells and kr in cells and \
                                paired_vs(cells[k], cells[kr], n_boot=4000)["beats_ref"]:
                            good.append(c)
                    if not good:
                        row += f"{'never':>12s}"
                    elif min(good) <= min(cs):
                        row += f"{'<=%.2f' % min(cs):>12s}"
                    else:
                        row += f"{'%.2f' % min(good):>12s}"
                print(row)
    print("\n=== NOISE vs DECOUPLING (is a wrong signal worse than a noisy one?) ===")
    print("normalized regret (1.00 = random floor) at c=1 (perfect direction, only "
          "noise) vs at c=0 (real but orthogonal signal)")
    for b in budgets:
        for a in allocs:
            row = f"  b={b:g} {a:26s}"
            for s in snrs:
                k1, k0 = f"{b}|{s}|{max(cs)}|{a}", f"{b}|{s}|0.0|{a}"
                kr = f"{b}|{s}|{max(cs)}|random"
                if k1 in cells and k0 in cells and kr in cells:
                    den = cells[kr]["mean_regret"]
                    row += (f"  SNR{s:g}: {cells[k1]['mean_regret']/den:.2f}"
                            f"/{cells[k0]['mean_regret']/den:.2f}")
            print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["part1", "f4", "map"])
    ap.add_argument("path")
    ap.add_argument("--snr", type=float, default=None)
    a = ap.parse_args()
    {"part1": part1, "f4": f4_table, "map": transfer_map}[a.cmd](a)


if __name__ == "__main__":
    main()
