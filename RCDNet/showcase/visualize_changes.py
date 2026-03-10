#!/usr/bin/env python3
"""
Visualize change detection results: Before | After | Change overlay
Output: one PNG grid per class in showcase/visualizations/
"""

import os
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "showcase" / "data"
RESULTS_DIR = ROOT / "showcase" / "results" / "change_maps"
OUT_DIR     = ROOT / "showcase" / "visualizations"


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def overlay_mask(img_rgb, mask, color=(1.0, 0.15, 0.15), alpha=0.45):
    out = img_rgb.copy()
    m = mask > 0
    for c, col in enumerate(color):
        out[m, c] = out[m, c] * (1 - alpha) + col * alpha
    return out


def top_patches(cls_dir, n, seed):
    """Return n patches sorted by number of changed pixels (most first)."""
    scored = []
    for p in cls_dir.glob("*.png"):
        arr = np.array(Image.open(p))
        scored.append((p.stem, int((arr > 0).sum())))
    scored.sort(key=lambda x: -x[1])
    if seed is not None:
        random.seed(seed)
    # take top 3*n then sample to add variety
    pool = [pid for pid, _ in scored[:max(n * 3, 20)]]
    return random.sample(pool, min(n, len(pool)))


def make_grid(cls_name, patch_ids, title_suffix=""):
    fig, axes = plt.subplots(len(patch_ids), 3,
                             figsize=(15, 5 * len(patch_ids)),
                             squeeze=False)

    cls_label = cls_name.replace("_", " ").title()
    cls_dir   = RESULTS_DIR / cls_name

    axes[0][0].set_title("Before (2021)",  fontsize=14, fontweight="bold", pad=10)
    axes[0][1].set_title("After  (2024)",  fontsize=14, fontweight="bold", pad=10)
    axes[0][2].set_title(f"Changes: {cls_label}", fontsize=14, fontweight="bold", pad=10)

    for row, pid in enumerate(patch_ids):
        img_a   = load_rgb(DATA_DIR / "A" / f"{pid}.png")
        img_b   = load_rgb(DATA_DIR / "B" / f"{pid}.png")
        mask    = np.array(Image.open(cls_dir / f"{pid}.png"))
        overlay = overlay_mask(img_b, mask)

        n_px  = int((mask > 0).sum())
        pct   = n_px / mask.size * 100

        axes[row][0].imshow(img_a)
        axes[row][1].imshow(img_b)
        axes[row][2].imshow(overlay)

        patch = mpatches.Patch(color=(1.0, 0.15, 0.15),
                               label=f"{n_px:,} px  ({pct:.1f}%)")
        axes[row][2].legend(handles=[patch], loc="lower right",
                            fontsize=9, framealpha=0.7, edgecolor="white")

        axes[row][0].set_ylabel(pid, fontsize=10, rotation=0,
                                labelpad=60, va="center", fontweight="bold")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(
        f"{cls_label}{title_suffix}\n"
        "Saint-Denis / Village Olympique  ·  2021 → 2024",
        fontsize=16, fontweight="bold", y=1.002
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize RCDNet change detection results")
    parser.add_argument("--class", dest="cls", default=None,
                        help="Class name to visualize (default: all classes)")
    parser.add_argument("--n", type=int, default=8,
                        help="Number of patches per grid (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    available = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print("Available classes:")
    for c in available:
        n = len(list((RESULTS_DIR / c).glob("*.png")))
        print(f"  {c}  ({n} detections)")
    print()

    classes = [args.cls] if args.cls else available

    for cls_name in classes:
        cls_dir = RESULTS_DIR / cls_name
        patch_ids = top_patches(cls_dir, args.n, args.seed)
        if not patch_ids:
            print(f"  [skip] {cls_name} — no detections")
            continue

        print(f"  Generating grid for {cls_name} ({len(patch_ids)} patches)...")
        fig = make_grid(cls_name, patch_ids)
        out = OUT_DIR / f"{cls_name}_grid.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out}")

    # --- All-classes best: one row per class, best patch each ---
    print("\n  Generating all-classes summary...")
    best_rows = []
    for cls_name in available:
        cls_dir = RESULTS_DIR / cls_name
        scored  = [(p.stem, int((np.array(Image.open(p)) > 0).sum()))
                   for p in cls_dir.glob("*.png")]
        if not scored:
            continue
        best_pid = max(scored, key=lambda x: x[1])[0]
        best_rows.append((cls_name, best_pid))

    if best_rows:
        fig2, axes2 = plt.subplots(len(best_rows), 3,
                                   figsize=(15, 5 * len(best_rows)),
                                   squeeze=False)
        axes2[0][0].set_title("Before (2021)",  fontsize=14, fontweight="bold", pad=10)
        axes2[0][1].set_title("After  (2024)",  fontsize=14, fontweight="bold", pad=10)
        axes2[0][2].set_title("Changes (overlay)", fontsize=14, fontweight="bold", pad=10)

        for row, (cls_name, pid) in enumerate(best_rows):
            img_a   = load_rgb(DATA_DIR / "A" / f"{pid}.png")
            img_b   = load_rgb(DATA_DIR / "B" / f"{pid}.png")
            mask    = np.array(Image.open(RESULTS_DIR / cls_name / f"{pid}.png"))
            overlay = overlay_mask(img_b, mask)

            n_px = int((mask > 0).sum())
            pct  = n_px / mask.size * 100
            label = cls_name.replace("_", " ").title()

            axes2[row][0].imshow(img_a)
            axes2[row][1].imshow(img_b)
            axes2[row][2].imshow(overlay)

            patch = mpatches.Patch(color=(1.0, 0.15, 0.15),
                                   label=f"{n_px:,} px  ({pct:.1f}%)")
            axes2[row][2].legend(handles=[patch], loc="lower right",
                                 fontsize=9, framealpha=0.7, edgecolor="white")
            axes2[row][0].set_ylabel(f"{label}\n{pid}", fontsize=10, rotation=0,
                                     labelpad=80, va="center", fontweight="bold")
            for ax in axes2[row]:
                ax.axis("off")

        plt.suptitle("Best detection per class — Saint-Denis / Village Olympique  ·  2021 → 2024",
                     fontsize=16, fontweight="bold", y=1.002)
        plt.tight_layout()
        out2 = OUT_DIR / "all_classes_best.png"
        fig2.savefig(out2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved → {out2}")

    print("\nDone. Open the PNGs in Windows Explorer:")
    print(f"  \\\\wsl$\\Ubuntu\\{str(OUT_DIR).lstrip('/')}")


if __name__ == "__main__":
    main()
