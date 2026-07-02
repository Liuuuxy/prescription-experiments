"""Build the daily PDF report (matplotlib PdfPages: auto-paginating text + embedded figures)."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({"font.family": "DejaVu Sans", "text.usetex": False})
NAVY = "#15315e"; GREY = "#333333"; ACCENT = "#0a6b3b"

# ---- content tokens: (style, text) ; styles: title, sub, h1, h2, body, bullet, num, space, image|caption ----
C = []
def T(t): C.append(("title", t))
def SUB(t): C.append(("sub", t))
def H1(t): C.append(("h1", t))
def H2(t): C.append(("h2", t))
def B(t): C.append(("body", t))
def L(t): C.append(("bullet", t))
def S(): C.append(("space", ""))
def IMG(path, cap): C.append(("image", (path, cap)))

# ============================ TITLE ============================
T("Targeted Data Selection & Gradient-Influence")
T("for Robot-Policy Fine-Tuning")
SUB("Daily research report  -  2026-06-30")
S()
B("One-line summary: we tested whether picking training demonstrations from the categories where a "
  "robot fails (\"smart\" selection) beats picking demos at random, and whether gradient-influence "
  "math can identify the most useful demos. Built a clean image-classification testbed to prove the "
  "method works in principle, diagnosed exactly why it underperforms on the robot, found one real "
  "improvement (whitening), and confirmed the central principle that governs when the whole approach "
  "can work.")
C.append(("pagebreak", ""))

# ============================ BACKGROUND ============================
H1("1.  The question, in plain language")
B("We are fine-tuning a robot vision-language-action policy (pi0) on a kitchen task: pick an object "
  "off a counter and place it in the sink (\"PickPlaceCounterToSink\"). We can only afford to train on "
  "about 200 demonstrations out of a pool of ~9,900. The core research question:")
L("Does choosing those 200 demos from the object categories the robot fails at (we call this the "
  "\"core\" or \"targeted\" arm) beat just choosing 200 at random?")
L("And can \"gradient influence\" - a math tool that scores each demo by how much it should help - "
  "pick the best demos automatically?")
S()
H2("Jargon, defined once (you don't need a math background)")
L("Gradient: the direction in which you'd nudge the model's settings to reduce its error on one "
  "example. Think of it as an arrow pointing toward \"learn this.\"")
L("Influence: if two examples' arrows point the same way, training on one should also help the other. "
  "So we score a candidate demo by how aligned its arrow is with the arrows of the demos we care about "
  "(the failures).")
L("Cosine similarity: a number from -1 to +1 measuring how aligned two arrows are, ignoring their "
  "length. +1 = same direction, 0 = unrelated, -1 = opposite.")
L("AUC: \"how well does a score separate the good items from the rest?\" 1.0 = perfect ranking, "
  "0.5 = no better than coin-flip. We use it to measure whether a selection score finds the right demos.")
L("Failure region / hard categories: the object types the robot is bad at (e.g., tall, thin items it "
  "can't grasp).")
S()
H1("2.  What we did today (overview)")
L("Built a clean testbed (image classification) where we KNOW the right answer, to check if "
  "gradient-influence works at all.  -> It works near-perfectly there.")
L("Diagnosed why it underperforms on the robot, using a geometry analysis of the gradients.  -> Found "
  "the governing principle.")
L("Worked through the \"does ignoring gradient size make sense?\" question with loss-surface reasoning.")
L("Tested three concrete improvements, each independently double-checked: whitening (a win), "
  "per-instance targeting (a dead end), and a controlled experiment that locks the principle.")
L("Finished the corrected RC-LESS selection experiment and the balanced evaluation, and produced "
  "category-level heatmaps of selection vs. outcome.")
C.append(("pagebreak", ""))

# ============================ CIFAR ============================
H1("3.  A clean testbed: does gradient-influence work at all?")
B("The robot results were ambiguous, so we built a controlled version where we know the correct answer "
  "in advance. We trained an image classifier (ResNet-18) from scratch on CIFAR-100 (100 object "
  "classes), but deliberately made 20 of the classes RARE in the training set (only 10% of their "
  "images). The model therefore becomes genuinely bad at those 20 classes - that is our \"failure "
  "region.\" Crucially, we now KNOW which extra images would help: the held-out images of those 20 rare "
  "classes. So we can finally grade the influence score against ground truth.")
S()
H2("Result: gradient-influence finds the helpful data almost perfectly")
L("AUC = 0.96-1.0 across training - the gradient score ranks the truly-helpful (rare-class) images "
  "right at the top.")
L("Interpretation: the method is NOT broken. When the thing you are targeting is clearly written into "
  "the gradient, influence works beautifully.")
L("Bonus finding: the signal stays strong (0.96) even late in training, because the rare classes stay "
  "un-learned. The signal only dies for things the model has already mastered.")
S()
B("This immediately reframes the robot result: if the same machinery scores 0.96 on a clean task but "
  "only ~0.60 on the robot, the problem is not the math - it is something about the robot task. The next "
  "section finds exactly what.")
S()
H1("4.  Why it underperforms on the robot: the gradient-geometry analysis")
B("We decomposed the cloud of per-demo gradient arrows into its principal directions (an SVD) and asked "
  "two questions: (a) is the cloud dominated by one \"everybody does this\" direction? and (b) how "
  "strongly does any direction line up with the thing we are targeting?")
S()
H2("The surprising, corrected finding")
L("It is NOT that the robot's gradients are low-dimensional or collapsed - they actually span MORE "
  "directions than CIFAR's. So our earlier \"common-mode collapse\" story was the wrong mechanism.")
L("The real difference is how strongly the gradient ENCODES the target distinction. Measured as the "
  "best-single-direction AUC for the target:")
B("        Setting                        gradient encodes target?      influence result")
B("        CIFAR class (rare)             strong  (best-dir 0.66)        AUC 0.96  (works)")
B("        Robot object-category          weak    (best-dir 0.56)        AUC 0.60  (weak)")
B("        CIFAR nuisance (brightness)    none    (best-dir 0.53)        AUC 0.50  (fails)")
S()
B("Why is object-category only weakly encoded on the robot? Because the robot's training loss is about "
  "predicting ACTIONS (reach, grasp, place), and those motions look similar across object types. The "
  "object's identity barely changes the gradient. So \"which object category\" is almost a nuisance "
  "variable as far as the action-prediction gradient is concerned - much like brightness is a nuisance "
  "for an image classifier.")
C.append(("pagebreak", ""))

# ============================ PRINCIPLE ============================
H1("5.  The governing principle (the key insight of the day)")
B("Putting the testbed and the geometry together gives a single, now-validated rule:")
S()
H2("Gradient-influence data selection works only if the loss gradient ENCODES the distinction you are "
   "targeting.")
S()
B("You can check this cheaply BEFORE running any expensive selection-and-training: compute the "
  "best-single-direction AUC of your target. If it is well above 0.5 (say >= 0.7), influence will help; "
  "if it is near 0.5, no scoring trick will rescue it. This explains every robot result we have: every "
  "selection method (targeted, contrast, influence, RC-LESS) tops out near 0.60 because they all target "
  "object-category, which the action loss does not strongly encode.")
S()
H1("6.  Does ignoring gradient SIZE make sense? (cosine vs. magnitude)")
B("Cosine similarity only uses the DIRECTION of the gradient arrow and throws away its LENGTH. Is that "
  "wise? We reasoned it through with simple loss-surface pictures and then checked it on data. The "
  "answer is: it depends on what the length means.")
L("If a long arrow means \"the model has a lot to learn here\" (informative) and a short arrow means "
  "\"already learned, redundant,\" then length is signal - keep it. Cosine alone would wrongly treat a "
  "redundant demo as good. (This is the trap on the robot.)")
L("If long vs. short is just per-example noise unrelated to helpfulness, then length is a distraction - "
  "dropping it (cosine) is fine or even better. (This is what we measured on CIFAR: cosine 0.958 vs. "
  "length-aware 0.937.)")
L("Practical takeaway: the safe default is the length-aware, optimizer-preconditioned score; but the "
  "real driver of success is still whether the target is encoded at all (Section 5), not this choice.")
C.append(("pagebreak", ""))

# ============================ THREE IMPROVEMENTS ============================
H1("7.  Three improvements we tested (each independently verified)")
H2("7a.  Whitening - a real win")
B("Idea in plain terms: before scoring, remove the handful of \"everyone does this\" directions that "
  "all demos share (the generic reach-grasp-place motion), so what remains is the failure-specific "
  "signal. Technically: project out the top-k shared principal directions, then score.")
L("Robot result: separation AUC improved from 0.605 to 0.677, and the share of genuinely-targeted demos "
  "in the top-200 picks doubled (17% -> 36%).")
L("Verified real, not an artifact: removing 50 RANDOM directions instead does nothing (stays 0.605 "
  "across 8 seeds); the removed directions carry 32% of the energy (real shared structure, not noise); "
  "shuffling the labels removes the lift. An independent agent re-derived every number from scratch.")
L("Caveat: you must not over-do it. On CIFAR, removing too many directions (k>30) destroys the signal "
  "itself. Use a bounded, modest k.")
S()
H2("7b.  Per-instance targeting - a dead end")
B("We tested whether targeting individual failed demos (instead of a category average) recovers more "
  "signal - via per-demo max, top-K, clustering, and nearest-neighbor scores.")
L("Best finer method reached 0.613 vs. the 0.607 baseline = +0.006, i.e. inside the noise. The single "
  "apparent \"win\" was just a lucky random seed.")
L("Conclusion: the robot signal is genuinely weak, not merely washed out by averaging. So this is not "
  "worth pursuing; it also reinforces the principle in Section 5.")
S()
H2("7c.  Controlled common-mode experiment - locks the principle")
B("On the SAME CIFAR gradients, we asked them to find targets the loss does and does not encode:")
L("Class (the trained target): best-direction AUC 0.66 - found well.")
L("Brightness / color / saturation (nuisances the loss never optimized): 0.53 - chance.")
L("Random label (pure control): 0.51 - chance.")
B("This is the controlled proof: identical gradients separate an encoded target but fail on un-encoded "
  "ones - exactly mirroring why robot object-category gives only weak influence.")
C.append(("pagebreak", ""))

# ============================ ROBOT RESULTS ============================
H1("8.  Robot fine-tuning results")
H2("Corrected RC-LESS selection")
B("RC-LESS is a gradient-influence selector. An earlier version scored 0.477 (worst of all) - we traced "
  "that to a bug in our own selector (a penalty term that steered selection AWAY from the failure "
  "region). The corrected version was retrained and evaluated today:")
L("RC-LESS v2 = 0.590 (177/300) - essentially tied with the targeted \"core\" arm (0.593). So the fix "
  "erased the regression; the method matches core but does not beat it.")
S()
H2("Balanced (stratified) evaluation - the powered comparison")
B("The overall 300-rollout eval has no statistical power for the targeted categories (they are a small "
  "slice of all categories). So we ran a balanced eval that focuses rollouts on the targeted categories "
  "(~280 rollouts per arm):")
B("        Arm                         success on the hard categories")
B("        baseline (no fine-tune)     0.262")
B("        random selection            0.351")
B("        core (targeted selection)   0.371")
S()
L("Fine-tuning clearly helps the hard categories: both random and core beat the no-fine-tune baseline "
  "by ~9-11 points (statistically significant, p = 0.02 and p = 0.005).")
L("But WHICH demos you pick does not matter: core beats random by only +0.020 (p = 0.62) - pure noise. "
  "Targeted selection ties random.")
S()
H2("A data-quality issue we found and fixed")
B("The original list of 10 \"hard\" categories was derived from tiny samples. On a balanced re-derivation, "
  "only 5 of the 10 survive as genuinely hard (kept: cheese_grater, pitcher, juice, jar, spray; dropped "
  "5 that were not actually hard). So the original targeting was diluted with easy categories - another "
  "reason targeted selection showed no edge.")
C.append(("pagebreak", ""))

# ============================ FIGURES ============================
H1("9.  Category-level heatmaps")
B("These visualize the results category-by-category. Read groups and patterns, not single noisy cells.")
IMG("/data/xinyua11/robocasa/weakregion/heatmap_stratified.png",
    "Fig 1. Balanced (high-power) eval on the 10 targeted categories. Rows ordered hardest-first; the "
    "5 dot-marked rows survived the balanced re-derivation and are the darkest (genuinely hard). "
    "core vs random differences are small and mixed -> net +0.02 (not significant).")
IMG("/data/xinyua11/robocasa/weakregion/heatmap_selection_vs_success.png",
    "Fig 2. LEFT = how many of each method's 200 demos came from each category (exact). RIGHT = success "
    "rate (overall eval, ~1-10 rollouts/cell, noisy). 'core' pours its budget into the hard categories "
    "(juice 29, spray 28, ...) yet does not reliably out-succeed random there - the left/right mismatch "
    "IS the 'targeting != better outcome' result.")
IMG("/data/xinyua11/robocasa/weakregion/heatmap_all_categories.png",
    "Fig 3. All 80 categories x 6 arms (overall eval). Bottom row = overall means. Targeting differences "
    "are invisible at the overall scale because ~70 easy categories dominate; the hard categories stay "
    "hard across every selection method.")

# ============================ INSIGHTS / NEXT ============================
H1("10.  Insights")
L("Gradient-influence is sound (CIFAR proves it: AUC 0.96). The robot under-performance is task-specific, "
  "not a flaw in the method.")
L("THE principle: influence selection works only if the loss gradient encodes your target. Cheap to "
  "check up front via best-direction AUC.")
L("Robot object-category is weakly encoded by the action-prediction loss, so every category-targeting "
  "selector ties random (~0.60 ranking; tie at the rollout level).")
L("Fine-tuning itself helps the hard categories a lot; the *choice* of demos (targeted vs random) does "
  "not - on the current target.")
L("Whitening is a genuine, verified improvement to the ranking (0.605 -> 0.677); per-instance targeting "
  "is not worth pursuing.")
S()
H1("11.  Next steps")
L("Change the TARGET, not the score. Point the influence at something the action loss DOES encode - "
  "e.g., grasp success/failure, the motion phase that fails, or specific scenes - rather than object "
  "category. The CIFAR bar to aim for is best-direction AUC >= 0.7.")
L("Adopt whitening (bounded k ~ 30-50) as the default robot selector and, if a target with stronger "
  "encoding is found, re-run select -> fine-tune -> balanced eval to test for an actual rollout win.")
L("Finish the magnitude menu (cosine vs. length-aware) on the robot gradients - currently extracting; "
  "it will confirm whether keeping gradient length recovers any of the weak signal.")
L("For fully trustworthy per-category outcomes, extend the balanced (high-power) eval beyond the 10 "
  "targeted categories so the success heatmap is as reliable as the selection heatmap.")
L("Optionally test a 'balanced-core' arm that selects on the corrected 5/10 hard categories - though, "
  "given targeted already ties random, it is expected to tie as well.")
S()
B("Artifacts produced today: CIFAR sandbox (xgrad.py, grad_geometry.py, cifar_control.py), whitening "
  "added to the robot scorer (influence_offline.py), per-instance analysis (rc_instance.py), three "
  "heatmaps, and this report. All findings above were independently re-derived by a verification agent.")

# ============================ RENDER ============================
STYLE = {
    "title": dict(size=18, weight="bold", color=NAVY, lh=0.034, gap=0.004, wrap=60, x=0.08),
    "sub":   dict(size=12, weight="normal", color=ACCENT, lh=0.028, gap=0.01, wrap=80, x=0.08),
    "h1":    dict(size=14, weight="bold", color=NAVY, lh=0.030, gap=0.018, wrap=80, x=0.07),
    "h2":    dict(size=11.5, weight="bold", color=GREY, lh=0.025, gap=0.012, wrap=92, x=0.07),
    "body":  dict(size=9.4, weight="normal", color=GREY, lh=0.0188, gap=0.004, wrap=104, x=0.08),
    "bullet":dict(size=9.4, weight="normal", color=GREY, lh=0.0188, gap=0.004, wrap=99, x=0.085),
    "space": dict(size=9, weight="normal", color=GREY, lh=0.012, gap=0.0, wrap=100, x=0.08),
}
TOP, BOT = 0.945, 0.06

def new_fig():
    fig = plt.figure(figsize=(8.5, 11)); return fig

pdf = PdfPages("/data/xinyua11/robocasa/weakregion/daily_report_2026-06-30.pdf")
fig = new_fig(); y = TOP; page = 1

def footer(fig, page):
    fig.text(0.5, 0.03, f"- {page} -", ha="center", fontsize=8, color="#999999")
    fig.text(0.08, 0.03, "pi0 / RoboCasa  -  gradient-influence data selection", fontsize=7, color="#bbbbbb")

for style, payload in C:
    if style == "pagebreak":
        footer(fig, page); pdf.savefig(fig); plt.close(fig)
        fig = new_fig(); y = TOP; page += 1; continue
    if style == "image":
        path, cap = payload
        footer(fig, page); pdf.savefig(fig); plt.close(fig); page += 1
        fig = new_fig()
        try:
            img = plt.imread(path)
            ax = fig.add_axes([0.06, 0.16, 0.88, 0.78]); ax.imshow(img); ax.axis("off")
        except Exception as e:
            fig.text(0.1, 0.5, f"[missing figure: {path}]", color="red")
        for k, ln in enumerate(textwrap.wrap(cap, 110)):
            fig.text(0.08, 0.13 - k*0.018, ln, fontsize=8.3, color=GREY, style="italic")
        footer(fig, page); pdf.savefig(fig); plt.close(fig)
        fig = new_fig(); y = TOP; page += 1; continue
    st = STYLE[style]
    if style == "space":
        y -= st["lh"]; continue
    y -= st["gap"]
    lines = textwrap.wrap(payload, st["wrap"]) or [""]
    is_bullet = style == "bullet"
    for i, ln in enumerate(lines):
        if y - st["lh"] < BOT:
            footer(fig, page); pdf.savefig(fig); plt.close(fig)
            fig = new_fig(); y = TOP; page += 1
        x = st["x"]
        txt = ln
        if is_bullet:
            if i == 0:
                txt = "•  " + ln
            else:
                x = st["x"] + 0.018
        fig.text(x, y, txt, fontsize=st["size"], weight=st["weight"], color=st["color"],
                 ha="left", va="top", parse_math=False)
        y -= st["lh"]

footer(fig, page); pdf.savefig(fig); plt.close(fig)
pdf.close()
print("wrote /data/xinyua11/robocasa/weakregion/daily_report_2026-06-30.pdf  (%d pages)" % page)
