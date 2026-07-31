"""
combine_real_vs_synthetic.py
============================
Crops the "Real" and "Synthetic" rows from the Kaggle-generated comparison
images and stacks them into a single, clean 3-row figure for the paper.

Produces:
  figures/fig6a_real_vs_synth_mnist.pdf / .png
  figures/fig6b_real_vs_synth_cifar.pdf / .png

Requirements:  pip install matplotlib Pillow numpy
Run from: Elsevier_Article__paper/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

BASE  = os.path.dirname(__file__)
SRC   = os.path.join(BASE, "figures", "real vs gen")
OUTDIR = os.path.join(BASE, "figures")

plt.rcParams.update({
    "font.family": "serif",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load(fname):
    return np.array(Image.open(os.path.join(SRC, fname)).convert("RGB"))

def crop_row(img_arr, row="real"):
    """
    Each Kaggle figure has exactly 2 rows of images:
      top half  → Real
      bottom half → Synthetic
    We crop a generous horizontal band for each, trimming the title area.
    """
    h, w, _ = img_arr.shape
    # Title text takes roughly the top 12% of the image
    title_frac = 0.12
    title_px   = int(h * title_frac)
    body       = img_arr[title_px:, :, :]     # strip title
    bh         = body.shape[0]
    # The bold label "Real X" / "Synthetic (Method)" sits above each row grid
    # Each half is approximately equal
    mid = bh // 2
    if row == "real":
        return body[:mid, :, :]
    else:
        return body[mid:, :, :]


def make_combined(real_arr, deep_arr, pca_arr,
                  row_labels, title, outname):
    """Stack 3 cropped row arrays vertically with row labels."""
    rows   = [real_arr, deep_arr, pca_arr]
    colors = ["#333333", "#CB181D", "#2171B5"]   # grey / red / blue

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5))
    for ax, row_img, label, color in zip(axes, rows, row_labels, colors):
        ax.imshow(row_img)
        ax.set_ylabel(label, fontsize=11, color=color, fontweight="bold",
                      rotation=0, labelpad=120, va="center", ha="right")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout(h_pad=0.3)

    out = os.path.join(OUTDIR, outname)
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"Saved: {out}")
    plt.close()


# ── MNIST ─────────────────────────────────────────────────────────────────────
print("Building MNIST combined figure …")
mnist_real  = crop_row(load("mnist real vs deepsmote 9.png"), "real")
mnist_deep  = crop_row(load("mnist real vs deepsmote 9.png"), "synth")
mnist_pca   = crop_row(load("mnist real vs pca smote 9.png"), "synth")

make_combined(
    mnist_real, mnist_deep, mnist_pca,
    row_labels=[
        "Real\n(Class 9, n=40)",
        "DeepSMOTE\nFID = 20.5",
        "PCA-SMOTE\nFID = 32.2",
    ],
    title="MNIST Class-9 (minority): real vs DeepSMOTE vs PCA-SMOTE synthetic images\n"
          "Despite higher FID, PCA-SMOTE achieves +5.75% higher minority accuracy (85.6% vs 79.9%)",
    outname="fig6a_real_vs_synth_mnist.pdf",
)

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
print("Building CIFAR combined figure …")
cifar_real  = crop_row(load("cifar real Vs deepsmote airplane.png"), "real")
cifar_deep  = crop_row(load("cifar real Vs deepsmote airplane.png"), "synth")
cifar_pca   = crop_row(load("cifar real vs pca smote airplane.png"), "synth")

make_combined(
    cifar_real, cifar_deep, cifar_pca,
    row_labels=[
        "Real\n(Airplane, n=80)",
        "DeepSMOTE",
        "PCA-SMOTE",
    ],
    title="CIFAR-10 Airplane (minority): real vs DeepSMOTE vs PCA-SMOTE synthetic images\n"
          "Both methods produce blurry 32×32 images; PCA-SMOTE better preserves structural identity",
    outname="fig6b_real_vs_synth_cifar.pdf",
)

print("\n✅ Done.")
