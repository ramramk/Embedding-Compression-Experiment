from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_results(metrics_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(metrics_csv)
    plots_dir = output_dir / "plots"
    hit_cols = ["hit@1", "hit@5", "hit@10"]
    method_markers = {
        "full": "X",
        "pca": "o",
        "truncation": "s",
        "int8": "^",
    }
    method_order = ["pca", "truncation", "int8", "full"]
    hit_colors = {
        "hit@1": "#1f77b4",
        "hit@5": "#ff7f0e",
        "hit@10": "#2ca02c",
    }

    for col in hit_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        multi_point_groups = []
        single_point_groups = []
        for method, group in df.groupby("method"):
            group = group.sort_values("embedding_dim")
            if len(group) == 1:
                single_point_groups.append((method, group))
            else:
                multi_point_groups.append((method, group))

        for method, group in multi_point_groups:
            ax.plot(
                group["embedding_dim"],
                group[col],
                marker=method_markers.get(method, "o"),
                label=method,
            )

        if len(single_point_groups) > 1:
            step = 10
            start = -step * (len(single_point_groups) - 1) / 2
            singleton_offsets = [start + i * step for i in range(len(single_point_groups))]
        else:
            singleton_offsets = [0]

        for (method, group), offset in zip(single_point_groups, singleton_offsets):
            x_plot = group["embedding_dim"].astype(float) + offset
            y_plot = group[col]
            ax.scatter(
                x_plot,
                y_plot,
                s=90,
                marker=method_markers.get(method, "X"),
                edgecolors="black",
                linewidths=0.8,
                label=f"{method} (single setting)",
                zorder=5,
            )
            ax.annotate(
                method,
                (float(x_plot.iloc[0]), float(y_plot.iloc[0])),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel(col.upper())
        ax.set_title(f"{col.upper()} vs embedding dimension")
        ax.legend()
        _save(fig, plots_dir / f"{col.replace('@', '_at_')}_vs_dimension.png")

    for x_col, name, xlabel in [
        ("bytes_per_vector", "hit_vs_memory_per_vector", "Estimated bytes per vector"),
        ("estimated_total_vector_bytes", "hit_vs_total_memory", "Estimated total source-vector bytes"),
        ("compression_ratio_vs_full", "compression_ratio_vs_hit", "Compression ratio vs full baseline"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        x_range = max(float(df[x_col].max() - df[x_col].min()), 1.0)
        method_offsets = {
            "pca": -0.012 * x_range,
            "truncation": 0.0,
            "int8": 0.012 * x_range,
            "full": 0.024 * x_range,
        }
        for col in hit_cols:
            for method in method_order:
                group = df[df["method"] == method]
                if group.empty:
                    continue
                ax.scatter(
                    group[x_col].astype(float) + method_offsets.get(method, 0.0),
                    group[col],
                    marker=method_markers.get(method, "o"),
                    color=hit_colors[col],
                    s=90 if method == "full" else 70,
                    edgecolors="black",
                    linewidths=0.8 if method == "full" else 0.5,
                    alpha=0.95 if method == "full" else 0.85,
                    zorder=4 if method == "full" else 3,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Hit rate")
        ax.set_title(xlabel + " vs Hit@k, color by k and shape by method")
        ax.text(
            0.01,
            0.01,
            "Points are slightly offset horizontally by method to reveal overlaps.",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
        hit_handles = [
            Line2D([0], [0], marker="o", color="w", label=col.upper(), markerfacecolor=color, markersize=8)
            for col, color in hit_colors.items()
        ]
        method_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=method,
                markerfacecolor="lightgray",
                markeredgecolor="black",
                markersize=8,
            )
            for method, marker in method_markers.items()
            if method in set(df["method"])
        ]
        first_legend = ax.legend(
            handles=hit_handles,
            title="Hit metric",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=method_handles,
            title="Method",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.58),
            borderaxespad=0.0,
        )
        _save(fig, plots_dir / f"{name}.png")

    baseline_like = df.sort_values(["method", "embedding_dim"]).drop_duplicates("method", keep="last")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(baseline_like))
    ax.bar([i - 0.25 for i in x], baseline_like["hit@1"], width=0.25, label="Hit@1")
    ax.bar(x, baseline_like["hit@5"], width=0.25, label="Hit@5")
    ax.bar([i + 0.25 for i in x], baseline_like["hit@10"], width=0.25, label="Hit@10")
    ax.set_xticks(list(x), baseline_like["method"])
    ax.set_ylabel("Hit rate")
    ax.set_title("Hit@k comparison by method at largest available dimension")
    ax.set_xlabel("Method, one condition per method")
    ax.legend()
    _save(fig, plots_dir / "method_hit_bar.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(df))
    ax.bar(positions, df["bytes_per_vector"])
    ax.set_xticks(list(positions), df["condition"], rotation=45, ha="right")
    ax.set_ylabel("Estimated bytes per vector")
    ax.set_title("Vector memory usage by condition")
    _save(fig, plots_dir / "vector_memory_usage_bar.png")

    for x_col, name, xlabel in [
        ("bytes_per_vector", "query_time_vs_memory_per_vector", "Estimated bytes per vector"),
        ("estimated_total_vector_bytes", "query_time_vs_total_memory", "Estimated total source-vector bytes"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for method in method_order:
            group = df[df["method"] == method].sort_values(x_col)
            if group.empty:
                continue
            ax.plot(
                group[x_col],
                group["query_time_seconds"],
                marker=method_markers.get(method, "o"),
                linewidth=1.5 if len(group) > 1 else 0,
                linestyle="-" if len(group) > 1 else "None",
                label=method,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    str(row["condition"]),
                    (float(row[x_col]), float(row["query_time_seconds"])),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Query time, seconds")
        ax.set_title(xlabel + " vs query time")
        ax.legend(title="Method")
        _save(fig, plots_dir / f"{name}.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    x_range = max(float(df["query_time_seconds"].max() - df["query_time_seconds"].min()), 1.0)
    method_offsets = {
        "pca": -0.012 * x_range,
        "truncation": 0.0,
        "int8": 0.012 * x_range,
        "full": 0.024 * x_range,
    }
    for col in hit_cols:
        for method in method_order:
            group = df[df["method"] == method]
            if group.empty:
                continue
            ax.scatter(
                group["query_time_seconds"].astype(float) + method_offsets.get(method, 0.0),
                group[col],
                marker=method_markers.get(method, "o"),
                color=hit_colors[col],
                s=90 if method == "full" else 70,
                edgecolors="black",
                linewidths=0.8 if method == "full" else 0.5,
                alpha=0.95 if method == "full" else 0.85,
                zorder=4 if method == "full" else 3,
            )
    ax.set_xlabel("Query time, seconds")
    ax.set_ylabel("Hit rate")
    ax.set_title("Latency vs Hit@k, color by k and shape by method")
    ax.text(
        0.01,
        0.01,
        "Points are slightly offset horizontally by method to reveal overlaps.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    hit_handles = [
        Line2D([0], [0], marker="o", color="w", label=col.upper(), markerfacecolor=color, markersize=8)
        for col, color in hit_colors.items()
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=method,
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
        )
        for method, marker in method_markers.items()
        if method in set(df["method"])
    ]
    first_legend = ax.legend(
        handles=hit_handles,
        title="Hit metric",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=method_handles,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.58),
        borderaxespad=0.0,
    )
    _save(fig, plots_dir / "latency_vs_hit.png")


def write_markdown_summary(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    best_h10 = df.sort_values("hit@10", ascending=False).iloc[0]
    smallest = df.sort_values("bytes_per_vector").iloc[0]
    lines = [
        "# PAN11 Retrieval Experiment Results",
        "",
        "This experiment evaluates source-document retrieval with chunked suspicious/source documents, Chroma vector search, and document-level aggregation of chunk evidence.",
        "",
        "## Best Hit@10",
        "",
        f"- Condition: {best_h10['condition']}",
        f"- Hit@10: {best_h10['hit@10']:.4f}",
        f"- Bytes per vector: {int(best_h10['bytes_per_vector'])}",
        "",
        "## Smallest Vector Representation",
        "",
        f"- Condition: {smallest['condition']}",
        f"- Bytes per vector: {int(smallest['bytes_per_vector'])}",
        f"- Hit@10: {smallest['hit@10']:.4f}",
        "",
        "## Full Metrics",
        "",
        df.to_markdown(index=False),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
