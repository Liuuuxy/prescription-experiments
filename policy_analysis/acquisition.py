"""Core algorithm: budgeted data-acquisition for imitation learning.

Given per-region estimates of (teacher success, student failure, student
uncertainty), score each region and allocate a fixed generation budget.

DERIVATION (why the multiplicative score):
  The real budget is SIM ATTEMPTS, not demos (collecting a success in region r
  costs ~1/P(teacher succeeds|r) attempts — measured: tall objects 38% => 2.6x).
    demos per attempt in r        = P(teacher succeeds | r)
    improvement per demo in r     ∝ P(student fails | r) * uncertainty(r)
  => expected improvement per ATTEMPT
       = P(teacher) * P(student_fails) * uncertainty  = the acquisition score.
  So allocating attempts ∝ score MAXIMIZES expected student improvement per unit
  generation cost. The product is derived, not heuristic.

NOISE ROBUSTNESS (from the n=50 lesson): per-region rates are noisy, so we score
with Wilson LOWER confidence bounds — be *confident* the teacher succeeds AND the
student fails before spending budget there. This stops the allocator from chasing
small-sample flukes.

UNCERTAINTY TERM is optional: the DP action-variance probe decides the algorithm's
form — if uncertainty just tracks failure (high AUC) it's redundant (use 2-term);
if it's orthogonal-but-real, keep it (3-term); if it's noise, drop it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wilson(k, n, z=1.96, bound="lower"):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) if bound == "lower" else (min(1.0, c + h) if bound == "upper" else p)


@dataclass
class Region:
    name: str
    teacher_k: int          # teacher successes
    teacher_n: int          # teacher attempts evaluated
    student_k: int          # student successes
    student_n: int          # student attempts evaluated
    uncertainty: float = 1.0  # student epistemic uncertainty (1.0 = unused)


def score_region(r: Region, robust=True, use_uncertainty=True):
    if robust:
        p_teacher = wilson(r.teacher_k, r.teacher_n, bound="lower")        # confident teacher CAN demo
        p_sfail = wilson(r.student_n - r.student_k, r.student_n, bound="lower")  # confident student fails
    else:
        p_teacher = r.teacher_k / max(r.teacher_n, 1)
        p_sfail = 1 - r.student_k / max(r.student_n, 1)
    unc = r.uncertainty if use_uncertainty else 1.0
    return p_teacher * p_sfail * unc


def allocate(regions, budget_attempts, robust=True, use_uncertainty=True, mode="proportional"):
    """Return per-region attempt allocation + expected demos/improvement."""
    scored = [(r, score_region(r, robust, use_uncertainty)) for r in regions]
    total = sum(s for _, s in scored) or 1.0
    out = []
    for r, s in sorted(scored, key=lambda x: -x[1]):
        if mode == "proportional":
            attempts = budget_attempts * s / total
        else:  # greedy top-1 gets all (degenerate; proportional is the default)
            attempts = budget_attempts if s == max(x[1] for x in scored) else 0
        p_teacher = (wilson(r.teacher_k, r.teacher_n, bound="lower") if robust
                     else r.teacher_k / max(r.teacher_n, 1))
        exp_demos = attempts * p_teacher
        out.append(dict(region=r.name, score=round(s, 4),
                        attempts=round(attempts, 1),
                        teacher_succ=round(r.teacher_k / max(r.teacher_n, 1), 2),
                        student_succ=round(r.student_k / max(r.student_n, 1), 2),
                        exp_demos=round(exp_demos, 1)))
    return out


def _print(plan, title):
    print(f"\n=== {title} ===")
    print(f"{'region':<16}{'score':>8}{'teach':>7}{'stud':>7}{'attempts':>10}{'exp_demos':>11}")
    for p in plan:
        print(f"{p['region']:<16}{p['score']:>8.3f}{p['teacher_succ']:>7.0%}"
              f"{p['student_succ']:>7.0%}{p['attempts']:>10.0f}{p['exp_demos']:>11.0f}")


def selftest_and_demo():
    print("SELF-TEST: a region where the TEACHER also fails should get ~0 budget")
    regs = [
        Region("student_fails_teacher_ok",  teacher_k=45, teacher_n=50, student_k=5,  student_n=50),  # ideal
        Region("both_fail (unteachable)",   teacher_k=3,  teacher_n=50, student_k=2,  student_n=50),  # avoid
        Region("student_ok (no need)",       teacher_k=48, teacher_n=50, student_k=46, student_n=50),  # skip
    ]
    _print(allocate(regs, budget_attempts=1000, use_uncertainty=False),
           "synthetic (budget=1000 attempts)")
    print("\n  -> the ideal region (student fails, teacher succeeds) gets the budget;")
    print("     'both_fail' is down-weighted (can't be demonstrated); 'student_ok' ~0.")

    # ---- real data: pi0 as TEACHER by height bin (n=150 weak-region run) ----
    # student profile is illustrative (DP per-region eval is a Phase-1 output);
    # uses the measured teacher rates short 67% / mid 55% / tall 36%.
    print("\nDEMO with REAL pi0 teacher rates (student = illustrative weak DP):")
    real = [
        Region("short(<6cm)",  teacher_k=33, teacher_n=49, student_k=12, student_n=40),
        Region("mid(6-11cm)",  teacher_k=28, teacher_n=51, student_k=8,  student_n=40),
        Region("tall(>11cm)",  teacher_k=18, teacher_n=50, student_k=2,  student_n=40),
    ]
    _print(allocate(real, budget_attempts=1000, use_uncertainty=False),
           "real teacher rates (budget=1000 attempts)")
    print("\n  -> tall is the student's weak region, but its LOW teacher rate (36%) tempers")
    print("     the budget vs a naive 'all-in-on-tall' plan — and the exp_demos column shows")
    print("     the real generation cost (fewer demos per attempt where the teacher struggles).")


if __name__ == "__main__":
    selftest_and_demo()
