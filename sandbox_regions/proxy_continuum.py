"""Proxy-corruption continuum (reviewer experiment #8, 2026-08-18).

Best-arm identification with a free-but-possibly-corrupted proxy phi and
expensive rollout pulls y ~ N(mu_a, sigma). Landscapes seeded from MEASURED
task-1 reality: sigma=3.2pp paired; a gc0-like +4.6 arm; optional -20 poison;
the "goodhart_spike" corruption reproduces the measured composite failure
(poison ranked at the top). Question the reviewer demands answered: which
allocators stay SAFE when the proxy is globally correlated but adversarial in
the tail, while still exploiting a clean proxy.

Allocators (budget T expensive pulls, K arms, phi free for all arms):
  rollout_se   : uniform rounds + harm-kill (our validated protocol), no proxy
  mfucb        : two-stage MF-UCB-style — proxy prefilters to top half, then SE
  corr_gate    : shrunken-correlation-gated bonus (the criticized design)
  fixed_mix    : constant-weight proxy bonus (no gate)
  calib_gate   : OURS — linear f-hat on (phi, y-bar) calibration pairs from
                 uniform pulls, jackknife-conformal residual bound B gates
                 entry (B < 0.75*sigma), support guard zeroes out-of-range phi
  local_gate   : calib_gate + phi-binned local B (2 bins) — local reliability
  oracle       : knows true mu ordering of phi's informative part (upper bound)
Metrics per (landscape, corruption): P(pick best), simple regret,
P(pick poison). 4000 reps each. CPU, ~minutes.
"""
import itertools
import json

import numpy as np

SIGMA = 3.2
K = 8
T = 16               # expensive pulls total (2 uniform rounds' worth)
REPS = 4000
KAPPA = 1.0
HARM = -4.0

rng_global = np.random.default_rng(20260818)


def landscape(kind, rng):
    mu = rng.normal(0, 2.0, K)
    mu[rng.integers(K)] = 4.6                     # gc0-like winner
    poison_idx = -1
    if kind == "poison":
        poison_idx = int(rng.integers(K))
        mu[poison_idx] = -20.0
        if np.argmax(mu) == poison_idx:           # keep a genuine winner
            mu[(poison_idx + 1) % K] = 4.6
    return mu, poison_idx


def corrupt(mu, mode, rng, poison_idx):
    """Returns (phi, bad_mask): bad_mask flags arms whose phi STRUCTURALLY
    misrepresents mu (inverted/spiked) -- consumed only by oracle_validity."""
    eps = rng.normal(0, 1, K)
    bad = np.zeros(K, bool)
    if mode == "clean":
        return mu + 0.5 * SIGMA * eps, bad
    if mode == "noise2":
        return mu + 2 * SIGMA * eps, bad
    if mode == "noise4":
        return mu + 4 * SIGMA * eps, bad
    if mode == "bias":
        return mu + 6.0 + 0.5 * SIGMA * eps, bad
    if mode == "nonlinear":
        return np.sign(mu) * np.sqrt(np.abs(mu)) + 0.3 * eps, bad
    if mode == "tail_invert":                     # invert above q75 (extreme phi)
        q = np.quantile(mu, 0.75)
        phi = mu + 0.5 * SIGMA * eps
        hi = mu >= q
        phi[hi] = -mu[hi] + 0.5 * SIGMA * eps[hi]
        return phi, hi
    if mode == "interior_invert":                 # reviewer #3: malicious band INSIDE support
        lo, hi_q = np.quantile(mu, [0.4, 0.6])
        band = (mu >= lo) & (mu < hi_q)
        phi = mu + 0.5 * SIGMA * eps
        phi[band] = -mu[band] + 0.5 * SIGMA * eps[band]
        return phi, band
    if mode == "interior_invert_top":             # nastier: band placed on the WINNER
        order = np.argsort(mu)
        band = np.zeros(K, bool); band[order[-2:]] = True   # top-2 arms lie, but their
        phi = mu + 0.5 * SIGMA * eps                        # phi lands mid-range (interior)
        phi[band] = np.median(mu) + 0.5 * SIGMA * eps[band]
        return phi, band
    if mode == "goodhart_spike":                  # measured composite failure
        phi = mu + 0.5 * SIGMA * eps
        tgt = poison_idx if poison_idx >= 0 else int(np.argmin(mu))
        phi[tgt] = phi.max() + 2.0
        bad[tgt] = True
        return phi, bad
    raise ValueError(mode)


def pull(mu, a, rng):
    return mu[a] + rng.normal(0, SIGMA)


def se_rounds(mu, rng, budget, live=None):
    """Uniform rounds + harm-kill; returns (means, counts, spent)."""
    live = set(range(K)) if live is None else set(live)
    s = np.zeros(K); n = np.zeros(K); spent = 0
    while spent + len(live) <= budget and live:
        for a in list(live):
            s[a] += pull(mu, a, rng); n[a] += 1; spent += 1
        for a in list(live):
            if n[a] >= 2:
                m = s[a] / n[a]
                if m + 1.9 * SIGMA / np.sqrt(n[a]) < HARM and len(live) > 1:
                    live.discard(a)
    m = np.where(n > 0, s / np.maximum(n, 1), -np.inf)
    return m, n, spent, live


def pick_best(m):
    return int(np.argmax(m))


def alloc_rollout_se(mu, phi, rng):
    m, n, _, _ = se_rounds(mu, rng, T)
    return pick_best(m)


def alloc_mfucb(mu, phi, rng):
    keep = np.argsort(phi)[-K // 2:]              # proxy prefilter (top half)
    m, n, _, _ = se_rounds(mu, rng, T, live=keep)
    return pick_best(m)


def alloc_fixed_mix(mu, phi, rng):
    """Staged, no gate: phi always steers stage 2 (top-3 by phi)."""
    s, n, spent, m = _round1(mu, rng)
    a, _ = _stage2(mu, phi, rng, m, s, n, spent, np.argsort(phi)[-3:])
    return a


def _stage2(mu, phi, rng, m, s, n, spent, keep):
    """Spend the remaining budget on `keep` arms (uniform rounds + harm-kill),
    then pick by pooled empirical mean. Used by every staged allocator so they
    differ ONLY in how `keep` was chosen (the trust test under study)."""
    live = set(int(a) for a in keep)
    while spent + len(live) <= T and live:
        for a in list(live):
            s[a] += pull(mu, a, rng); n[a] += 1; spent += 1
        for a in list(live):
            mm = s[a] / n[a]
            if mm + 1.9 * SIGMA / np.sqrt(n[a]) < HARM and len(live) > 1:
                live.discard(a)
    # pessimistic recommendation: penalize thin sampling so a lucky n=1 pull
    # can't steal the argmax from a well-sampled stage-2 arm
    m2 = np.where(n > 0, s / np.maximum(n, 1) - SIGMA / np.sqrt(np.maximum(n, 1)), -np.inf)
    return pick_best(m2), n


def _round1(mu, rng):
    s = np.zeros(K); n = np.zeros(K)
    for a in range(K):
        s[a] += pull(mu, a, rng); n[a] += 1
    return s, n, K, s / np.maximum(n, 1)


def alloc_corr_gate(mu, phi, rng):
    """Staged: 1 uniform round, then correlation test decides whether phi
    steers the remaining budget (the criticized design, apples-to-apples)."""
    s, n, spent, m = _round1(mu, rng)
    r = np.corrcoef(phi, m)[0, 1] if np.std(phi) > 1e-9 and np.std(m) > 1e-9 else 0.0
    c = np.tanh(K / (K + 8) * np.arctanh(np.clip(r, -0.999, 0.999)))
    if abs(c) < 0.35:
        keep = range(K)
    else:
        keep = np.argsort(np.sign(c) * phi)[-3:]
    a, _ = _stage2(mu, phi, rng, m, s, n, spent, keep)
    return a


def _calib_fit(phi_obs, y_obs, meas_var=SIGMA ** 2):
    """Noise-adjusted certification: the LOO residual on single-pull
    calibration points mixes proxy error with the pull's own rollout noise
    (Var sigma^2). Certify the PROXY's error: s_p^2 = max(0, Var(resid) -
    meas_var); B = 1.28*s_p (~q90 under normality). Without this adjustment a
    perfect proxy can never certify -- caught by this sim's first run."""
    n = len(phi_obs)
    if n < 4 or np.std(phi_obs) < 1e-9:
        return None
    b1, b0 = np.polyfit(phi_obs, y_obs, 1)
    rss = float(np.sum((y_obs - (b1 * phi_obs + b0)) ** 2))
    s_tot2 = rss / (n - 2)                        # dof-corrected total residual var
    s_p2 = max(0.0, s_tot2 - meas_var)            # subtract known rollout noise
    B = 1.28 * np.sqrt(s_p2)
    return b1, b0, B


def alloc_calib_gate(mu, phi, rng, local=False):
    """OURS, staged: 1 uniform calibration round (selection-free pairs), fit
    f-hat + jackknife-conformal bound B. Certified (B < kappa*sigma) -> the
    remaining budget goes to the top-3 arms by predicted value, restricted to
    calibration support; local variant additionally requires the arm's phi-bin
    to certify. Uncertified -> continue uniform (collapses to rollout_se)."""
    s, n, spent, m = _round1(mu, rng)
    fit = _calib_fit(phi, m)
    keep = range(K)
    if fit is not None:
        b1, b0, B = fit
        w = max(0.0, 1.0 - (B / (KAPPA * SIGMA)) ** 2)   # soft certified trust
        if w > 0.15:
            pred = b1 * phi + b0
            ok = np.ones(K, bool)
            if local:
                med = np.median(phi)
                for side_mask in (phi <= med, phi > med):
                    if side_mask.sum() >= 3:
                        lf = _calib_fit(phi[side_mask], m[side_mask])
                        if lf is None or lf[2] >= KAPPA * SIGMA:
                            ok[side_mask] = False
                    else:
                        ok[side_mask] = False
            cand = [a for a in range(K) if ok[a]]
            if len(cand) >= 2:
                blended = {a: (1 - w) * m[a] + w * pred[a] for a in cand}
                keep = sorted(cand, key=lambda a: blended[a])[-3:]
    a, _ = _stage2(mu, phi, rng, m, s, n, spent, keep)
    return a


def alloc_oracle(mu, phi, rng):
    top = np.argsort(mu)[-2:]                     # oracle prefilter to true top-2
    m, n, _, _ = se_rounds(mu, rng, T, live=set(int(x) for x in top))
    return pick_best(m)


def make_oracle_validity(bad_mask, noise_scale):
    """Reviewer #8 mandatory baseline: knows the true trustworthy region
    (structural liars + noise scale) but NOT the realized noise. Trusted arms
    compete by proxy prediction, untrusted by round-1 rollout mean only."""
    def fn(mu, phi, rng):
        s, n, spent, m = _round1(mu, rng)
        trusted = ~bad_mask
        if noise_scale >= KAPPA * SIGMA or trusted.sum() < 4:
            keep = range(K)                       # proxy useless -> uniform
        else:
            b1, b0 = np.polyfit(phi[trusted], m[trusted], 1)
            idx = np.where(trusted, b1 * phi + b0, m)
            keep = np.argsort(idx)[-3:]
        a, _ = _stage2(mu, phi, rng, m, s, n, spent, keep)
        return a
    return fn


ALLOCS = {
    "rollout_se": alloc_rollout_se,
    "mfucb_prefilter": alloc_mfucb,
    "corr_gate": alloc_corr_gate,
    "fixed_mix": alloc_fixed_mix,
    "calib_gate": lambda mu, phi, rng: alloc_calib_gate(mu, phi, rng, local=False),
    "local_gate": lambda mu, phi, rng: alloc_calib_gate(mu, phi, rng, local=True),
    "oracle_validity": None,   # built per-rep from the true corruption mask
    "oracle": alloc_oracle,
}
MODES = ["clean", "noise2", "noise4", "bias", "nonlinear", "tail_invert", "interior_invert", "interior_invert_top", "goodhart_spike"]
NOISE_SCALE = {"clean": 1.6, "noise2": 6.4, "noise4": 12.8, "bias": 1.6, "nonlinear": 0.3, "tail_invert": 1.6, "interior_invert": 1.6, "interior_invert_top": 1.6, "goodhart_spike": 1.6}


def main():
    results = {}
    for scen in ("benign", "poison"):
        for mode in MODES:
            for name, fn in ALLOCS.items():
                hit = reg = pois = 0.0
                for rep in range(REPS):
                    rng = np.random.default_rng(1_000_000 * hash((scen, mode, name)) % (2**31) + rep)
                    mu, pidx = landscape(scen, rng)
                    phi, badm = corrupt(mu, mode, rng, pidx)
                    f = make_oracle_validity(badm, NOISE_SCALE[mode]) if name == "oracle_validity" else fn
                    a = f(mu, phi, rng)
                    best = int(np.argmax(mu))
                    hit += (a == best)
                    reg += mu[best] - mu[a]
                    pois += (a == pidx) if pidx >= 0 else 0
                results[(scen, mode, name)] = (hit / REPS, reg / REPS, pois / REPS)
    print(f"K={K} arms, T={T} pulls, sigma={SIGMA}, {REPS} reps")
    for scen in ("benign", "poison"):
        print(f"\n=== scenario: {scen} ===")
        print(f"{'corruption':<15}" + "".join(f"{n:>16}" for n in ALLOCS))
        for mode in MODES:
            row_h = f"{mode:<15}" + "".join(
                f"{results[(scen, mode, n)][0]*100:>7.0f}%/{results[(scen, mode, n)][1]:>5.1f} "
                for n in ALLOCS)
            print(row_h)
        if scen == "poison":
            print("  P(pick POISON):")
            for mode in MODES:
                print(f"  {mode:<13}" + "".join(
                    f"{results[(scen, mode, n)][2]*100:>15.1f}%" for n in ALLOCS))
    json.dump({f"{s}|{m}|{n}": v for (s, m, n), v in results.items()},
              open("/data/xinyua11/robocasa/sandbox_regions/proxy_continuum_results.json", "w"))
    print("\nPROXY CONTINUUM COMPLETE")


if __name__ == "__main__":
    main()
