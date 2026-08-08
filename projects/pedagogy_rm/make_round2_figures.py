"""Draw the four figures for the second round of work.

    python projects/pedagogy_rm/make_round2_figures.py

Each of these exists because a table of the same numbers made a reader do arithmetic to see the
point. Nothing here is decorative: if a figure did not change what someone would do next, it is
not in this file.

  lr_efficiency   the learning-rate sweep. The point is that the ranking by reward and the
                  ranking by reward-per-drift are DIFFERENT orderings, which a table hides.
  rubric_v2       V1 against V2 agreement, dimension by dimension, with the 0.4 gate drawn.
  decoupling      why V1's five dimensions were really three, as two correlation matrices.
  leak_repairs    two failed repairs and one measurement, on one axis. A negative result.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    os.makedirs(args.figures, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25})

    # ---- 1. The learning-rate sweep, both orderings side by side ------------------------------
    rates = ["1e-5", "2e-5", "4e-5", "8e-5"]
    gain = [0.348, 0.526, 0.526, 0.606]
    kl = [0.080, 0.074, 0.122, 0.172]
    per_kl = [g / k for g, k in zip(gain, kl, strict=True)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    x = np.arange(len(rates))
    left.bar(x, gain, 0.6, color="#bbbbbb", edgecolor="#888888")
    left.set_xticks(x)
    left.set_xticklabels(rates)
    left.set_ylabel("reward gained over 35 steps")
    left.set_title("Ranked by reward: bigger step always wins", fontsize=10)
    for i, v in enumerate(gain):
        left.text(i, v + 0.012, f"{v:+.3f}", ha="center", fontsize=8.5)
    left.set_ylim(0, max(gain) * 1.18)

    colours = ["#999999", "#2f6db5", "#999999", "#c0392b"]
    right.bar(x, per_kl, 0.6, color=colours, edgecolor="#666666")
    right.set_xticks(x)
    right.set_xticklabels(rates)
    right.set_ylabel("reward gained per unit of KL drift")
    right.set_title("Ranked by efficiency: a different order", fontsize=10)
    for i, v in enumerate(per_kl):
        right.text(i, v + 0.13, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold" if i == 1 else "normal")
    right.set_ylim(0, max(per_kl) * 1.2)
    right.annotate(
        "chosen",
        (1, per_kl[1]),
        textcoords="offset points",
        xytext=(0, 26),
        ha="center",
        fontsize=8.5,
        color="#2f6db5",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#2f6db5"},
    )
    right.annotate(
        "arm B's regime:\nmore reward, worse rate",
        (3, per_kl[3]),
        textcoords="offset points",
        xytext=(-4, 34),
        ha="center",
        fontsize=8,
        color="#c0392b",
        arrowprops={"arrowstyle": "->", "color": "#c0392b"},
    )
    fig.suptitle("Learning rate: the two rankings disagree", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "lr_efficiency.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 2. V1 against V2 agreement ----------------------------------------------------------
    # Paired where a V2 dimension replaced a V1 one; the two new dimensions have no V1 partner.
    pairs = [
        ("hands_over\n(was elicits)", 0.56, 0.81),
        ("leak", 0.51, 0.83),
        ("correct", 0.18, 0.80),
        ("locates\n(was targeted)", 0.57, 0.67),
        ("verdict\n(new)", None, 0.73),
        ("guidance\n(new)", None, 0.51),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    xs = np.arange(len(pairs))
    v1 = [p[1] if p[1] is not None else 0 for p in pairs]
    v2 = [p[2] for p in pairs]
    ax.bar(xs - 0.2, v1, 0.4, label="first rubric", color="#bbbbbb", edgecolor="#888888")
    ax.bar(xs + 0.2, v2, 0.4, label="second rubric", color="#2f6db5", edgecolor="#1a3f6b")
    ax.axhline(0.4, color="#c0392b", lw=1.6, ls="--")
    # Placed in the empty region under the two new dimensions, where the first rubric has no
    # bar - on the line itself it collided with both the line and the guidance bar.
    ax.text(
        4.55,
        0.24,
        "0.4 — below this a\ndimension is discarded",
        fontsize=8,
        color="#c0392b",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#c0392b",
            "alpha": 0.95,
            "linewidth": 0.8,
        },
    )
    for i, (_, a, b) in enumerate(pairs):
        if a is not None:
            ax.text(i - 0.2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8)
        else:
            ax.text(i - 0.2, 0.015, "n/a", ha="center", fontsize=7.5, color="#888888")
        ax.text(i + 0.2, b + 0.015, f"{b:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8.5)
    ax.set_ylabel("agreement between raters (weighted kappa)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Every dimension now clears the gate; two of the old six did not", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "rubric_v2.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 3. Why V1 was really three dimensions -----------------------------------------------
    d1 = ["leak", "targeted", "actionable", "elicits", "correct"]
    m1 = np.array(
        [
            [1.00, -0.45, 0.34, 0.44, 0.35],
            [-0.45, 1.00, 0.35, 0.32, -0.06],
            [0.34, 0.35, 1.00, 0.95, 0.21],
            [0.44, 0.32, 0.95, 1.00, 0.23],
            [0.35, -0.06, 0.21, 0.23, 1.00],
        ]
    )
    d2 = ["guidance", "locates", "hands_over", "verdict", "leak", "correct"]
    m2 = np.array(
        [
            [1.00, 0.07, -0.24, 0.14, -0.44, 0.62],
            [0.07, 1.00, 0.12, 0.17, -0.09, -0.09],
            [-0.24, 0.12, 1.00, -0.17, 0.64, -0.17],
            [0.14, 0.17, -0.17, 1.00, -0.19, 0.22],
            [-0.44, -0.09, 0.64, -0.19, 1.00, -0.17],
            [0.62, -0.09, -0.17, 0.22, -0.17, 1.00],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, (names, m, title, eff) in zip(
        axes,
        (
            (d1, m1, "First rubric: 2.9 effective of 5", "actionable/elicits = 0.95"),
            (d2, m2, "Second rubric: 4.2 effective of 6", "worst pair = 0.64"),
        ),
        strict=True,
    ):
        im = ax.imshow(np.abs(m), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    hot = abs(m[i, j]) > 0.6
                    ax.text(
                        j,
                        i,
                        f"{m[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if abs(m[i, j]) > 0.55 else "#333",
                        fontweight="bold" if hot else "normal",
                    )
        ax.set_title(f"{title}\n{eff}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.7, label="|correlation| between dimensions")
    fig.suptitle("Rating one dimension at a time decoupled the scales", fontsize=12.5, fontweight="bold", y=1.0)
    fig.savefig(os.path.join(args.figures, "decoupling.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 4. The two failed repairs ------------------------------------------------------------
    labels = ["the human\n(3625 labels)", "head as fitted", "rank-1\ncorrection", "residualise\nbefore fitting"]
    vals = [0.43, 0.70, 0.67, 0.74]
    cols = ["#4c9f70", "#999999", "#e8a33d", "#c0392b"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    bars = ax.bar(range(len(vals)), vals, 0.6, color=cols, edgecolor="#555555")
    ax.axhline(0.43, color="#4c9f70", lw=1.6, ls="--")
    # In the gap between bars: on either bar top it collides with the value label.
    ax.text(1.5, 0.465, "what the human does", fontsize=8.5, color="#2c6b4a", ha="center")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold" if i == 3 else "normal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("how strongly leaking tracks length")
    ax.set_ylim(0, 1.14)
    ax.set_title("Two attempts to decouple leak from length; the second made it worse", fontsize=10.5)
    # Above the bar rather than beside it: at 0.74 the bar fills the plot and any offset
    # to the side lands on top of it.
    ax.annotate(
        "overfit: alpha fell to 0.1,\nspread rose 0.46 to 0.64",
        (3, 0.755),
        textcoords="offset points",
        xytext=(-118, 30),
        ha="center",
        fontsize=8,
        color="#c0392b",
        arrowprops={"arrowstyle": "->", "color": "#c0392b"},
    )
    bars[0].set_hatch("//")
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "leak_repairs.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for name in ("lr_efficiency", "rubric_v2", "decoupling", "leak_repairs"):
        print(f"wrote {args.figures}/{name}.png")


if __name__ == "__main__":
    main()
