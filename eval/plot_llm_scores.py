#!/usr/bin/env python3
"""
Generate plots from eval/llm_scores.jsonl and eval/llm_stats.json.
Creates grouped bar charts for per-topic overall scores and aggregate means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import statistics

import matplotlib.pyplot as plt


MODELS = ["baseline0", "baseline1", "baseline2", "got"]
COLORS = {
    "baseline0": "#1f77b4",
    "baseline1": "#ff7f0e",
    "baseline2": "#2ca02c",
    "got": "#d62728",
}


def load_scores(scores_path: Path) -> List[Dict]:
    return [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stats(stats_path: Path) -> Dict:
    return json.loads(stats_path.read_text(encoding="utf-8"))


def plot_per_topic_overall(records: List[Dict], out_path: Path) -> None:
    # Now uses the single metric 'ai_likeness' produced by eval_llm.py
    topics = [rec["topic"] for rec in records]
    x = range(len(topics))
    width = 0.18

    plt.figure(figsize=(max(10, len(topics) * 0.6), 8))
    for i, model in enumerate(MODELS):
        offsets = [xi + (i - 1.5) * width for xi in x]
        values = [rec["scores"].get(model, {}).get("ai_likeness", 0) for rec in records]
        plt.bar(offsets, values, width, label=model, color=COLORS.get(model))

    plt.xticks(x, topics, rotation=45, ha="right", fontsize=9)
    plt.ylabel("AI-likeness (1-10)")
    plt.ylim(0, 10)
    plt.title("AI-likeness by Topic and Model (1 = human-like, 10 = AI-like)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_stats(stats: Dict, records: List[Dict], out_path_raw: Path) -> None:
    # stats is expected to contain per-model means for the single metric 'ai_likeness'
    metric = "ai_likeness"

    # Compute per-model lists from records to get standard deviation
    models = MODELS
    per_model_values = {m: [] for m in models}
    for rec in records:
        for m in models:
            v = rec.get("scores", {}).get(m, {}).get(metric)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            per_model_values[m].append(fv)

    means = [ (stats.get(m, {}).get("raw_mean", {}).get(metric, 0)) for m in models ]
    stds = []
    for m in models:
        vals = per_model_values.get(m, [])
        if len(vals) >= 2:
            stds.append(statistics.stdev(vals))
        else:
            stds.append(0.0)

    plt.figure(figsize=(8, 6))
    x = range(len(models))
    bars = plt.bar(x, means, yerr=stds, capsize=6, color=[COLORS.get(m) for m in models])
    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Mean AI-likeness (1-10)")
    plt.ylim(0, 10)
    plt.title("Mean AI-likeness by Model (error bars = std dev)")

    # Annotate bars
    for bar, val, sd in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.2f}\n±{sd:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path_raw.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_raw, dpi=200)
    plt.close()


def main() -> None:
    scores_path = Path("eval/llm_scores.jsonl")
    stats_path = Path("eval/llm_stats.json")
    if not scores_path.exists() or not stats_path.exists():
        raise SystemExit("Missing llm_scores.jsonl or llm_stats.json in eval/")

    records = load_scores(scores_path)
    stats = load_stats(stats_path)

    plots_dir = Path("eval/plots")
    plot_per_topic_overall(records, plots_dir / "overall_scores_by_topic.png")
    plot_mean_stats(stats, records, plots_dir / "mean_raw_scores.png")
    print(f"Saved plots under {plots_dir}")


if __name__ == "__main__":
    main()
