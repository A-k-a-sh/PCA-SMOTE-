"""
generate_figures.py
====================
Run this script locally (not on Kaggle) to generate all paper figures.
Output: saves publication-quality PNGs into the figures/ subfolder.

Requirements:
    pip install matplotlib numpy

Usage:
    python generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})

BLUE   = "#2171B5"   # PCA-SMOTE
RED    = "#CB181D"   # DeepSMOTE
GREY   = "#636363"   # Baseline
GREEN  = "#238B45"   # Pixel SMOTE
ORANGE = "#D94801"   # 3W / 4W


# =============================================================================
# DATA — all numbers extracted from result files
# =============================================================================

# ── MNIST (Seed 10 target sweep) ─────────────────────────────────────────────
mnist_targets      = [100, 200, 500, 1000, 2000, 4000]
mnist_baseline_9   = 87.61   # seed 10

# First sweep block in mnist_pca_weighted_res.md = generate_synthetic_for_all_classes (DeepSMOTE)
mnist_deep_9       = [87.71, 87.61, 87.81, 87.12, 83.75, 79.98]

# Second sweep block = generate_synthetic_pcasmote (PCA-SMOTE)
mnist_pca_9        = [88.01, 88.90, 88.60, 87.22, 87.41, 82.85]

# ── CIFAR-10 (Seed 60 target sweep) ──────────────────────────────────────────
cifar_targets      = [100, 200, 500, 1000, 2000]
cifar_baseline_air = 24.90   # seed 60

cifar_deep_air     = [22.90, 23.50, 23.60, 20.20, 18.40]
cifar_pca_air      = [20.80, 25.10, 26.90, 20.40, 17.70]

# ── Multi-seed summary (seeds 10, 20, 60) — Digit-9 / Airplane minority class ──
seeds = [10, 20, 60]

# MNIST — per-seed, per-phase (class-9 acc %)
mnist_multiseed = {
    "Baseline":        [87.61, 87.12, 88.60],
    "Pixel SMOTE":     [80.18, 81.07, 78.39],
    "DeepSMOTE":       [79.88, 84.14, 82.36],
    "PCA-SMOTE (t=4000)":   [85.63, 87.12, 87.02],
    # optimal target per seed (200, 200, 500)
    "PCA-SMOTE (opt)": [88.90, 88.90, 88.60],
}

# CIFAR — per-seed, per-phase (airplane acc %)
cifar_multiseed = {
    "Baseline":        [23.50, 18.30, 24.90],
    "Pixel SMOTE":     [20.90, 16.60, 20.50],
    "DeepSMOTE":       [19.90, 14.80, 14.20],
    "PCA-SMOTE (t=2000)":   [22.20, 17.10, 20.40],  # Phase 4 main comparison
    "PCA-SMOTE (opt)": [22.20, 25.10, 26.90],  # best from sweep (t=100, 200, 500)
}

# ── FID / LPIPS vs Accuracy (MNIST, seed 10) ──────────────────────────────────
fid_methods   = ["DeepSMOTE", "PCA-SMOTE"]
fid_values    = [20.47,  32.25]     # FID — lower is "better image quality"
acc_values    = [79.88,  85.63]     # digit-9 accuracy at t=4000, seed 10


# =============================================================================
# FIGURE 1: Target Sweep (the "money figure") — MNIST + CIFAR side by side
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── subplot (a): MNIST ────────────────────────────────────────────────────────
ax = axes[0]
ax.axhline(mnist_baseline_9, color=GREY, linewidth=1.8,
           linestyle="--", label=f"Baseline (no aug.) = {mnist_baseline_9:.1f}%", zorder=1)
ax.plot(mnist_targets, mnist_deep_9, color=RED,  marker="s",
        linestyle="-",  label="DeepSMOTE", zorder=3)
ax.plot(mnist_targets, mnist_pca_9,  color=BLUE, marker="o",
        linestyle="-",  label="PCA-SMOTE (Ours)", zorder=4)

ax.fill_between(mnist_targets, mnist_deep_9, mnist_baseline_9,
                where=[p < mnist_baseline_9 for p in mnist_deep_9],
                alpha=0.10, color=RED, label="_nolegend_")
ax.fill_between(mnist_targets, mnist_pca_9, mnist_baseline_9,
                where=[p > mnist_baseline_9 for p in mnist_pca_9],
                alpha=0.12, color=BLUE, label="_nolegend_")

ax.set_xscale("log")
ax.set_xticks(mnist_targets)
ax.set_xticklabels([str(t) for t in mnist_targets])
ax.set_xlabel("Synthetic target count per class")
ax.set_ylabel("Class-9 (minority) accuracy (%)")
ax.set_title("(a) MNIST  [minority n=40, ratio 100:1]")
ax.set_ylim(76, 92)
ax.legend(frameon=False, loc="lower left")
ax.annotate("Synthetic\noverfitting", xy=(2000, 83.75), xytext=(800, 78.5),
            fontsize=9.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate("Peak\n(t=200)", xy=(200, 88.90), xytext=(350, 90.5),
            fontsize=9.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

# ── subplot (b): CIFAR ────────────────────────────────────────────────────────
ax = axes[1]
ax.axhline(cifar_baseline_air, color=GREY, linewidth=1.8,
           linestyle="--", label=f"Baseline (no aug.) = {cifar_baseline_air:.1f}%", zorder=1)
ax.plot(cifar_targets, cifar_deep_air, color=RED,  marker="s",
        linestyle="-",  label="DeepSMOTE", zorder=3)
ax.plot(cifar_targets, cifar_pca_air,  color=BLUE, marker="o",
        linestyle="-",  label="PCA-SMOTE (Ours)", zorder=4)

ax.fill_between(cifar_targets, cifar_deep_air, cifar_baseline_air,
                where=[p < cifar_baseline_air for p in cifar_deep_air],
                alpha=0.10, color=RED, label="_nolegend_")
ax.fill_between(cifar_targets, cifar_pca_air, cifar_baseline_air,
                where=[p > cifar_baseline_air for p in cifar_pca_air],
                alpha=0.12, color=BLUE, label="_nolegend_")

ax.set_xscale("log")
ax.set_xticks(cifar_targets)
ax.set_xticklabels([str(t) for t in cifar_targets])
ax.set_xlabel("Synthetic target count per class")
ax.set_ylabel("Airplane (minority) accuracy (%)")
ax.set_title("(b) CIFAR-10  [minority n=80, ratio 25:1]")
ax.set_ylim(12, 32)
ax.legend(frameon=False, loc="upper right")
ax.annotate("Synthetic\noverfitting", xy=(1000, 20.20), xytext=(300, 14.5),
            fontsize=9.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.annotate("Peak\n(t=500)", xy=(500, 26.90), xytext=(700, 29.0),
            fontsize=9.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

plt.tight_layout()
out = os.path.join(OUTDIR, "fig1_target_sweep.pdf")
plt.savefig(out)
plt.savefig(out.replace(".pdf", ".png"))
print(f"Saved: {out}")
plt.close()


# =============================================================================
# FIGURE 2: FID vs Accuracy Paradox (bar + scatter combo)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: FID bars (lower = better looking images)
ax = axes[0]
bars = ax.bar(fid_methods, fid_values, color=[RED, BLUE], alpha=0.85, edgecolor="white", width=0.5)
ax.set_ylabel("FID  (lower → more realistic images)")
ax.set_title("(a) Image Quality (FID)")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=11)
ax.set_ylim(0, 45)
for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

# Right: Accuracy bars (higher = better classifier)
ax = axes[1]
bars2 = ax.bar(fid_methods, acc_values, color=[RED, BLUE], alpha=0.85, edgecolor="white", width=0.5)
ax.set_ylabel("Class-9 accuracy (%)")
ax.set_title("(b) Classifier Utility (accuracy)")
ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=11)
ax.set_ylim(70, 92)
for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

# Add annotation arrows showing the paradox
axes[0].annotate("DeepSMOTE:\nBetter images\n(lower FID ✓)",
                 xy=(0, 20.47), xytext=(0.3, 35), fontsize=9, color=RED,
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
axes[1].annotate("PCA-SMOTE:\nBetter classifier\n(higher acc ✓)",
                 xy=(1, 85.63), xytext=(0.6, 88.5), fontsize=9, color=BLUE,
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

plt.suptitle("FID–Accuracy Paradox: better image quality ≠ better classifier\n"
             "(MNIST, Seed 10, target=4000)", fontsize=11, y=1.02)
plt.tight_layout()
out = os.path.join(OUTDIR, "fig2_fid_paradox.pdf")
plt.savefig(out)
plt.savefig(out.replace(".pdf", ".png"))
print(f"Saved: {out}")
plt.close()


# =============================================================================
# FIGURE 3: Multi-seed comparison bar chart (MNIST + CIFAR)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(seeds))
n_methods = 4  # we show: Baseline, DeepSMOTE, PCA-SMOTE(t=max), PCA-SMOTE(opt)
width = 0.18
colors = [GREY, RED, BLUE + "aa", BLUE]  # last blue is solid PCA optimal

def plot_multiseed_bars(ax, data, seeds, title, ylabel):
    keys   = ["Baseline", "DeepSMOTE", "PCA-SMOTE (t=4000)", "PCA-SMOTE (opt)"]
    clrs   = [GREY, RED, "#6BAED6", BLUE]
    labels = ["Baseline", "DeepSMOTE (full balance)", "PCA-SMOTE (full balance)", "PCA-SMOTE (optimal target)"]
    for i, (key, clr, lbl) in enumerate(zip(keys, clrs, labels)):
        offset = (i - (n_methods - 1) / 2) * width
        vals = data[key]
        bars = ax.bar(x + offset, vals, width, color=clr, alpha=0.88,
                      label=lbl, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

plot_multiseed_bars(
    axes[0], mnist_multiseed, seeds,
    "(a) MNIST — Digit-9 (minority) Accuracy",
    "Class-9 accuracy (%)"
)
axes[0].set_ylim(70, 95)

# For CIFAR, replace t=4000 key with t=2000 (max we tested)
cifar_plot = {
    "Baseline":              cifar_multiseed["Baseline"],
    "DeepSMOTE":             cifar_multiseed["DeepSMOTE"],
    "PCA-SMOTE (t=4000)":    cifar_multiseed["PCA-SMOTE (t=2000)"],
    "PCA-SMOTE (opt)":       cifar_multiseed["PCA-SMOTE (opt)"],
}
plot_multiseed_bars(
    axes[1], cifar_plot, seeds,
    "(b) CIFAR-10 — Airplane (minority) Accuracy",
    "Airplane accuracy (%)"
)
axes[1].set_ylim(8, 34)

plt.tight_layout()
out = os.path.join(OUTDIR, "fig3_multiseed_bars.pdf")
plt.savefig(out)
plt.savefig(out.replace(".pdf", ".png"))
print(f"Saved: {out}")
plt.close()


# =============================================================================
# FIGURE 4: PCA-SMOTE pipeline diagram (schematic)
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis("off")

# Box positions: x-center, y-center, width, height
boxes = [
    (0.06,  0.5, 0.10, 0.55, "Real images\n(minority class,\nn=40 to 80)",   "#F7FBFF"),
    (0.21,  0.5, 0.10, 0.55, "Encoder\n(DCGAN)\n→ z ∈ ℝ⁶⁰⁰",                "#DEEBF7"),
    (0.38,  0.5, 0.12, 0.55, "PCA\n(k≤20 dims)\nIntrinsic subspace",         "#9ECAE1"),
    (0.55,  0.5, 0.12, 0.55, "SMOTE\n(k=5 neighbours)\nin low-dim space",     "#4292C6"),
    (0.72,  0.5, 0.12, 0.55, "PCA⁻¹\nProject back\nto ℝ⁶⁰⁰",                "#2171B5"),
    (0.90,  0.5, 0.10, 0.55, "Decoder\n→ synthetic\nimage",                    "#08519C"),
]

for (cx, cy, w, h, label, color) in boxes:
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2, edgecolor="#555", facecolor=color
    )
    ax.add_patch(rect)
    txt_color = "white" if color in ("#2171B5", "#08519C", "#4292C6") else "#222"
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5,
            color=txt_color, multialignment="center")

# Arrows between boxes
arrow_xs = [0.115, 0.265, 0.445, 0.615, 0.785]
for ax_x in arrow_xs:
    ax.annotate("", xy=(ax_x + 0.01, 0.5), xytext=(ax_x - 0.01, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("PCA-SMOTE Pipeline: SMOTE is performed in the intrinsic low-dimensional subspace "
             "of each minority class, then projected back to the full latent space for decoding.",
             fontsize=10.5, pad=10)

out = os.path.join(OUTDIR, "fig4_pcasmote_pipeline.pdf")
plt.savefig(out)
plt.savefig(out.replace(".pdf", ".png"))
print(f"Saved: {out}")
plt.close()


# =============================================================================
# FIGURE 5: Nearest-Neighbour Distance Ratio (Latent Space Sparsity Diagnostic)
# =============================================================================
# NN ratio = mean(kNN distance) / mean(vector norm)
# Ratio → 1.0  means nearest neighbours are as far as random points → no structure
# Ratio < 0.9  means coherent local clusters exist

# Data from multi-seed experiments (deepsmote so far.md Section 8)
classes_mnist   = ["Class 0\n(n=4000)", "Class 1\n(n=2000)", "Class 7\n(n=100)",
                   "Class 8\n(n=60)",   "Class 9\n(n=40, minority)"]
nn_ratio_mnist  = [0.846, 0.853, 0.901, 0.940, 0.989]   # approx from seed 10 output

# CIFAR — both configs, mean over 3 seeds
classes_cifar   = ["Ship\n(n=2000,\nmajority)", "Bird\n(n=600)", "Dog\n(n=1000)",
                   "Dog\n(n=80,\nConfig A\nminority)", "Airplane\n(n=80,\nConfig B\nminority)"]
nn_ratio_cifar  = [0.824,  0.870, 0.861, 0.986, 0.956]   # means from deepsmote so far.md

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

THRESHOLD = 0.95  # above this → insufficient density for SMOTE

def plot_nn_ratio(ax, classes, ratios, title):
    colors = [RED if r >= THRESHOLD else BLUE for r in ratios]
    bars = ax.bar(classes, ratios, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {THRESHOLD} (SMOTE unreliable above)")
    ax.axhline(1.0, color="#999", linestyle=":", linewidth=1.2, label="Ratio = 1.0 (fully random)")
    ax.set_ylim(0.78, 1.04)
    ax.set_ylabel("NN Distance Ratio\n(higher → more sparse, less structured)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    # Annotate the problem bar
    for i, (cls, r) in enumerate(zip(classes, ratios)):
        if r >= THRESHOLD:
            ax.text(i, r + 0.003, f"{r:.3f}", ha="center", va="bottom",
                    fontsize=9, color=RED, fontweight="bold")
        else:
            ax.text(i, r + 0.003, f"{r:.3f}", ha="center", va="bottom",
                    fontsize=9, color="#333")

plot_nn_ratio(axes[0], classes_mnist,
              nn_ratio_mnist,
              "(a) MNIST — NN ratio by class\n(sorted by n)")
plot_nn_ratio(axes[1], classes_cifar,
              nn_ratio_cifar,
              "(b) CIFAR-10 — NN ratio by class\n(Config A & B minority highlighted)")

# Shared annotation
fig.text(0.5, -0.02,
         "Red bars exceed the NN ratio threshold (≥0.95), indicating insufficient\n"
         "latent density for meaningful SMOTE interpolation.",
         ha="center", fontsize=10.5, color="#333")

plt.tight_layout()
out = os.path.join(OUTDIR, "fig5_nn_ratio.pdf")
plt.savefig(out)
plt.savefig(out.replace(".pdf", ".png"))
print(f"Saved: {out}")
plt.close()


print("\n✅ All figures generated successfully!")
print(f"   Output folder: {OUTDIR}")
print("   Files:")
for f in sorted(os.listdir(OUTDIR)):
    if os.path.isfile(os.path.join(OUTDIR, f)):
        print(f"     {f}")
