from __future__ import annotations

import argparse
from pathlib import Path

from pan11_retrieval.plotting import plot_results, write_markdown_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots and Markdown summary from metrics CSV.")
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plot_results(args.metrics_csv, args.output_dir)
    write_markdown_summary(args.metrics_csv, args.output_dir / "RESULTS.md")


if __name__ == "__main__":
    main()

