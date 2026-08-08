#!/usr/bin/env python3
"""Aggregate Engram task-vector sweep JSON files into plots and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("dense", "engram")
ARM_LABELS = {"dense": "Dense", "engram": "Engram"}
COLORS = {"dense": "tab:orange", "engram": "tab:blue"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_reports(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        paths = sorted(input_dir.glob(f"{arm}_step*.json"))
        if not paths:
            raise FileNotFoundError(f"No {arm} reports found under {input_dir}.")
        reports[arm] = sorted([json.loads(path.read_text()) for path in paths], key=lambda row: row["source_step"])
    return reports


def task_accuracy(metrics: dict[str, Any]) -> float:
    return 100.0 * metrics["test_task"]["accuracy"]


def gain_recovery(report: dict[str, Any], accuracy: float) -> float:
    base = task_accuracy(report["final_before"])
    direct = task_accuracy(report["final_after_direct"])
    return 100.0 * (accuracy - base) / (direct - base)


def best_transfer(report: dict[str, Any]) -> dict[str, Any]:
    return report["best_task_vector"]


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_by_checkpoint(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, arm in zip(axes, ARMS):
        rows = reports[arm]
        steps = [row["source_step"] for row in rows]
        axis.plot(
            steps, [task_accuracy(row["source_before"]) for row in rows], marker="o", label="Early checkpoint, untuned"
        )
        axis.plot(
            steps,
            [task_accuracy(row["source_after"]) for row in rows],
            marker="o",
            label="Early checkpoint, post-trained",
        )
        axis.plot(
            steps,
            [task_accuracy(row["final_before"]) for row in rows],
            linestyle="--",
            label="Final checkpoint, untuned",
        )
        axis.plot(
            steps,
            [task_accuracy(row["final_after_direct"]) for row in rows],
            linestyle="--",
            label="Final checkpoint, directly post-trained",
        )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Source pretraining checkpoint (step)")
        axis.set_ylabel("Held-out routing accuracy (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Accuracy before and after post-training across checkpoint age")
    save_figure(fig, output_dir / "accuracy_by_source_checkpoint.png")


def plot_alpha_sweep(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, arm in zip(axes, ARMS):
        for report in reports[arm]:
            rows = report["task_vector_sweep"]
            axis.plot(
                [row["alpha"] for row in rows],
                [task_accuracy(row["metrics"]) for row in rows],
                marker="o",
                label=f"source step {report['source_step']}",
            )
        axis.axhline(
            task_accuracy(reports[arm][0]["final_after_direct"]),
            color="black",
            linestyle="--",
            linewidth=1,
            label="Direct final post-training",
        )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Task-vector scale α")
        axis.set_ylabel("Held-out routing accuracy (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Transferred accuracy as a function of α and checkpoint age")
    save_figure(fig, output_dir / "transfer_accuracy_vs_alpha.png")


def plot_alpha_heatmaps(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, arm in zip(axes, ARMS):
        rows = reports[arm]
        alphas = [item["alpha"] for item in rows[0]["task_vector_sweep"]]
        matrix = np.array([[task_accuracy(item["metrics"]) for item in row["task_vector_sweep"]] for row in rows])
        image = axis.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=100, cmap="viridis")
        axis.set_xticks(range(len(alphas)), [str(value) for value in alphas])
        axis.set_yticks(range(len(rows)), [str(row["source_step"]) for row in rows])
        axis.set_xlabel("Task-vector scale α")
        axis.set_ylabel("Source checkpoint step")
        axis.set_title(ARM_LABELS[arm])
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                axis.text(
                    x,
                    y,
                    f"{matrix[y, x]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[y, x] < 55 else "black",
                )
        fig.colorbar(image, ax=axis, label="Held-out accuracy (%)")
    fig.suptitle("Task-vector transfer accuracy heatmap")
    save_figure(fig, output_dir / "transfer_accuracy_heatmap.png")


def plot_best_transfer_by_age(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for arm in ARMS:
        rows = reports[arm]
        ages = [row["checkpoint_age_steps"] for row in rows]
        accuracies = [task_accuracy(best_transfer(row)["metrics"]) for row in rows]
        recoveries = [gain_recovery(row, accuracy) for row, accuracy in zip(rows, accuracies)]
        axes[0].plot(ages, accuracies, marker="o", label=ARM_LABELS[arm], color=COLORS[arm])
        axes[1].plot(ages, recoveries, marker="o", label=ARM_LABELS[arm], color=COLORS[arm])
    axes[0].set_title("Best transferred accuracy")
    axes[0].set_xlabel("Checkpoint age gap (pretraining steps)")
    axes[0].set_ylabel("Held-out routing accuracy (%)")
    axes[1].set_title("Direct-fine-tuning gain recovered")
    axes[1].set_xlabel("Checkpoint age gap (pretraining steps)")
    axes[1].set_ylabel("Accuracy gain recovery (%)")
    axes[1].axhline(90, color="black", linestyle="--", linewidth=1, label="90% threshold")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Transfer fidelity as the source checkpoint gets earlier")
    save_figure(fig, output_dir / "best_transfer_vs_checkpoint_age.png")


def plot_interpolation(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, arm in zip(axes, ARMS):
        for report in reports[arm]:
            rows = report["interpolation_sweep"]
            axis.plot(
                [row["lambda"] for row in rows],
                [task_accuracy(row["metrics"]) for row in rows],
                marker="o",
                label=f"source step {report['source_step']}",
            )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Early-post-trained weight λ")
        axis.set_ylabel("Held-out routing accuracy (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("Ordinary weight interpolation with the final checkpoint")
    save_figure(fig, output_dir / "interpolation_accuracy_vs_lambda.png")


def plot_repair(
    reports: dict[str, list[dict[str, Any]]], output_dir: Path, mode: str, title: str, filename: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, arm in zip(axes, ARMS):
        for report in reports[arm]:
            curve = report["repair"][mode]["curve"]
            axis.plot(
                [point["repair_steps"] for point in curve],
                [task_accuracy(point) for point in curve],
                marker="o",
                label=f"source step {report['source_step']}",
            )
        axis.set_title(ARM_LABELS[arm])
        axis.set_xlabel("Additional repair fine-tuning steps")
        axis.set_ylabel("Held-out routing accuracy (%)")
        axis.set_ylim(-2, 102)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(title)
    save_figure(fig, output_dir / filename)


def plot_forgetting(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    metric_specs = (
        ("climbmix_loss_relative_change", "ClimbMix loss change (%)"),
        ("wikitext_loss_relative_change", "WikiText-103 loss change (%)"),
        ("lambada_accuracy_change", "LAMBADA accuracy change (points)"),
    )
    for axis, (metric, title) in zip(axes, metric_specs):
        for arm in ARMS:
            rows = reports[arm]
            values = []
            for report in rows:
                value = best_transfer(report)["forgetting_vs_final"][metric]
                values.append(100.0 * value)
            axis.plot(
                [row["checkpoint_age_steps"] for row in rows],
                values,
                marker="o",
                label=ARM_LABELS[arm],
                color=COLORS[arm],
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Checkpoint age gap (steps)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Catastrophic-forgetting checks for selected task-vector transfers")
    save_figure(fig, output_dir / "catastrophic_forgetting.png")


def plot_forgetting_by_variant(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    metric_specs = (
        ("climbmix_loss_relative_change", "ClimbMix loss change (%)"),
        ("wikitext_loss_relative_change", "WikiText-103 loss change (%)"),
        ("lambada_accuracy_change", "LAMBADA accuracy change (points)"),
    )
    variants = (
        "Early direct post-train",
        "Final direct post-train",
        "Task-vector transfer",
        "Weight interpolation",
        "Task vector + 80 repair",
        "Interpolation + 80 repair",
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex="col")
    for arm_index, arm in enumerate(ARMS):
        rows = reports[arm]
        steps = [row["source_step"] for row in rows]
        variant_rows: dict[str, list[dict[str, float]]] = {variant: [] for variant in variants}
        for report in rows:
            variant_rows["Early direct post-train"].append(
                relative_forgetting_for_plot(report["source_before"]["general"], report["source_after"]["general"])
            )
            variant_rows["Final direct post-train"].append(
                relative_forgetting_for_plot(
                    report["final_before"]["general"], report["final_after_direct"]["general"]
                )
            )
            variant_rows["Task-vector transfer"].append(report["best_task_vector"]["forgetting_vs_final"])
            variant_rows["Weight interpolation"].append(report["best_interpolation"]["forgetting_vs_final"])
            variant_rows["Task vector + 80 repair"].append(
                report["repair"]["task_vector"]["curve"][-1]["forgetting_vs_final"]
            )
            variant_rows["Interpolation + 80 repair"].append(
                report["repair"]["interpolation"]["curve"][-1]["forgetting_vs_final"]
            )
        for metric_index, (metric, title) in enumerate(metric_specs):
            axis = axes[arm_index, metric_index]
            for variant in variants:
                axis.plot(
                    steps,
                    [100.0 * row[metric] for row in variant_rows[variant]],
                    marker="o",
                    markersize=3,
                    label=variant,
                )
            axis.axhline(0, color="black", linewidth=1)
            axis.set_title(f"{ARM_LABELS[arm]} — {title}")
            axis.set_xlabel("Source checkpoint step")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
            if metric_index == 2:
                axis.legend(fontsize=7, loc="best")
    fig.suptitle("General-capability change by transfer, merge, and repair variant")
    save_figure(fig, output_dir / "catastrophic_forgetting_by_variant.png")


def relative_forgetting_for_plot(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        "climbmix_loss_relative_change": (candidate["climbmix_loss"] - baseline["climbmix_loss"])
        / baseline["climbmix_loss"],
        "wikitext_loss_relative_change": (candidate["wikitext_loss"] - baseline["wikitext_loss"])
        / baseline["wikitext_loss"],
        "lambada_accuracy_change": (candidate["lambada_accuracy"] - baseline["lambada_accuracy"]),
    }


def write_summary_csv(reports: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fields = (
        "arm",
        "source_step",
        "target_step",
        "checkpoint_age_steps",
        "early_initial_accuracy",
        "early_posttrained_accuracy",
        "final_initial_accuracy",
        "final_direct_accuracy",
        "best_alpha",
        "best_transfer_accuracy",
        "accuracy_gain_recovery",
        "climbmix_loss_relative_change",
        "wikitext_loss_relative_change",
        "lambada_accuracy_change",
        "best_lambda",
        "best_interpolation_accuracy",
        "repaired_accuracy_80_steps",
    )
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for arm in ARMS:
            for report in reports[arm]:
                transfer = best_transfer(report)
                transfer_accuracy = task_accuracy(transfer["metrics"])
                repair_curve = report["repair"]["task_vector"]["curve"]
                interpolation = report["best_interpolation"]
                writer.writerow(
                    {
                        "arm": arm,
                        "source_step": report["source_step"],
                        "target_step": report["target_step"],
                        "checkpoint_age_steps": report["checkpoint_age_steps"],
                        "early_initial_accuracy": task_accuracy(report["source_before"]),
                        "early_posttrained_accuracy": task_accuracy(report["source_after"]),
                        "final_initial_accuracy": task_accuracy(report["final_before"]),
                        "final_direct_accuracy": task_accuracy(report["final_after_direct"]),
                        "best_alpha": transfer["alpha"],
                        "best_transfer_accuracy": transfer_accuracy,
                        "accuracy_gain_recovery": gain_recovery(report, transfer_accuracy),
                        "climbmix_loss_relative_change": transfer["forgetting_vs_final"][
                            "climbmix_loss_relative_change"
                        ],
                        "wikitext_loss_relative_change": transfer["forgetting_vs_final"][
                            "wikitext_loss_relative_change"
                        ],
                        "lambada_accuracy_change": transfer["forgetting_vs_final"]["lambada_accuracy_change"],
                        "best_lambda": interpolation["lambda"],
                        "best_interpolation_accuracy": task_accuracy(interpolation["metrics"]),
                        "repaired_accuracy_80_steps": task_accuracy(repair_curve[-1]),
                    }
                )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(args.input_dir)
    plot_accuracy_by_checkpoint(reports, args.output_dir)
    plot_alpha_sweep(reports, args.output_dir)
    plot_alpha_heatmaps(reports, args.output_dir)
    plot_best_transfer_by_age(reports, args.output_dir)
    plot_interpolation(reports, args.output_dir)
    plot_repair(
        reports,
        args.output_dir,
        mode="task_vector",
        title="Task-vector transfer followed by low-LR repair",
        filename="repair_accuracy_curves.png",
    )
    plot_repair(
        reports,
        args.output_dir,
        mode="interpolation",
        title="Best interior weight merge followed by low-LR repair",
        filename="interpolation_repair_accuracy_curves.png",
    )
    plot_forgetting(reports, args.output_dir)
    plot_forgetting_by_variant(reports, args.output_dir)
    write_summary_csv(reports, args.output_dir)
    (args.output_dir / "combined_results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"wrote plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
