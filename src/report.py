"""Generate summary reports and plots from evaluation results."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_report(results_dirs: list[str], output_dir: str):
    """
    Aggregate results from multiple experiment evaluation directories and
    produce a comparison report with plots.

    Each results_dir should contain a summary.json file.
    """
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    for d in results_dirs:
        summary_path = os.path.join(d, "summary.json")
        if os.path.exists(summary_path):
            s = load_summary(summary_path)
            s["eval_dir"] = d
            summaries.append(s)

    if not summaries:
        print("No summaries found. Nothing to report.")
        return

    df = pd.DataFrame(summaries)
    df["label"] = df.apply(_make_label, axis=1)
    df["train_label"] = df.apply(_make_train_label, axis=1)

    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")

    _print_table(df)
    _plot_accuracy_bar(df, output_dir)
    _plot_accuracy_grouped(df, output_dir)
    _plot_eval_template_comparison(df, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


def _make_train_label(row: pd.Series) -> str:
    """Build a label describing the training setup."""
    method = row.get("train_method", "?").upper()
    fmt = "chat" if row.get("train_chat_template") else "raw"
    parts = [method, f"train:{fmt}"]
    if row.get("train_mask_question"):
        parts.append("mask_q")
    return " | ".join(parts)


def _make_label(row: pd.Series) -> str:
    """Build a full label: training setup + eval format."""
    train = _make_train_label(row)
    eval_fmt = "chat" if row.get("eval_chat_template") else "raw"
    return f"{train} | eval:{eval_fmt}"


def _print_table(df: pd.DataFrame):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"{'Experiment':<55} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-'*80}")
    for _, row in df.iterrows():
        print(f"{row['label']:<55} {row['accuracy']:>10.2%} {row['correct']:>10} {row['total_samples']:>8}")
    print(f"{'='*80}\n")


def _plot_accuracy_bar(df: pd.DataFrame, output_dir: str):
    """Simple bar chart of all experiments."""
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.5), 6))
    colors = plt.cm.tab10.colors
    bars = ax.bar(range(len(df)), df["accuracy"], color=[colors[i % len(colors)] for i in range(len(df))])

    for bar, acc in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Evaluation Accuracy Across All Experiments")
    ax.set_ylim(0, min(1.0, df["accuracy"].max() + 0.15))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_bar.png"), dpi=150)
    plt.close(fig)


def _plot_accuracy_grouped(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: training config on x-axis, eval template as groups."""
    fig, ax = plt.subplots(figsize=(12, 6))

    train_labels = list(df["train_label"].unique())
    x = range(len(train_labels))
    width = 0.35

    chat_accs = []
    raw_accs = []
    for tl in train_labels:
        subset = df[df["train_label"] == tl]
        chat_row = subset[subset["eval_chat_template"] == True]
        raw_row = subset[subset["eval_chat_template"] == False]
        chat_accs.append(chat_row["accuracy"].values[0] if len(chat_row) > 0 else 0)
        raw_accs.append(raw_row["accuracy"].values[0] if len(raw_row) > 0 else 0)

    bars1 = ax.bar([i - width/2 for i in x], chat_accs, width, label="Eval: with chat template", color="#4C72B0")
    bars2 = ax.bar([i + width/2 for i in x], raw_accs, width, label="Eval: without chat template", color="#DD8452")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(train_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Effect of Eval Chat Template per Training Config")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_grouped.png"), dpi=150)
    plt.close(fig)


def _plot_eval_template_comparison(df: pd.DataFrame, output_dir: str):
    """Scatter plot: eval with chat template (x) vs eval without (y) for each training config."""
    train_labels = list(df["train_label"].unique())
    paired = {}
    for tl in train_labels:
        subset = df[df["train_label"] == tl]
        chat_row = subset[subset["eval_chat_template"] == True]
        raw_row = subset[subset["eval_chat_template"] == False]
        if len(chat_row) > 0 and len(raw_row) > 0:
            paired[tl] = {
                "chat": chat_row["accuracy"].values[0],
                "raw": raw_row["accuracy"].values[0],
            }

    if not paired:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for label, vals in paired.items():
        ax.scatter(vals["chat"], vals["raw"], s=100, zorder=5)
        ax.annotate(label, (vals["chat"], vals["raw"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    lim = [0, 1]
    ax.plot(lim, lim, "--", color="gray", alpha=0.5, label="y=x (equal)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Accuracy (eval WITH chat template)")
    ax.set_ylabel("Accuracy (eval WITHOUT chat template)")
    ax.set_title("Chat Template Effect on Evaluation")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eval_template_scatter.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison report from eval results")
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="Directories containing summary.json files")
    parser.add_argument("--output_dir", type=str, default="results/report",
                        help="Where to save the report")
    args = parser.parse_args()

    generate_report(args.results_dirs, args.output_dir)
