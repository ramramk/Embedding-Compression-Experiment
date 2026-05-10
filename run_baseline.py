from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from pan11_retrieval.config import ChunkConfig, ExperimentConfig
from pan11_retrieval.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PAN11 full-embedding baseline retrieval.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/baseline"), type=Path)
    parser.add_argument("--chroma-dir", default=Path("chroma_db/baseline"), type=Path)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--debug-source-limit", type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--rebuild-chroma", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        chroma_dir=args.chroma_dir,
        model_name=args.model_name,
        debug_limit=args.debug_limit,
        debug_source_limit=args.debug_source_limit,
        sample_fraction=args.sample_fraction,
        reuse_existing_chroma=not args.rebuild_chroma,
    )
    config = replace(config, chunking=ChunkConfig(args.chunk_size, args.overlap))
    run_experiment(config, baseline_only=True)


if __name__ == "__main__":
    main()
