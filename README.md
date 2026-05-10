# PAN11 Source Retrieval with Vector Compression

This project implements a local Python experiment for source-document retrieval on the PAN-PC-11 plagiarism dataset. It is intentionally scoped to retrieval: given a suspicious document, rank source documents and count success when at least one true source appears in the top-k.

## Experiment Design

The pipeline uses chunk-level retrieval and document-level aggregation:

1. Load source documents, suspicious documents, and PAN-style XML ground truth.
2. Chunk source and suspicious documents with a configurable character window and overlap.
3. Embed source chunks with SentenceTransformers.
4. Transform vectors for each condition: full float32, PCA, truncation, or scalar int8 quantization.
5. Store transformed source chunk vectors in a persistent Chroma collection.
6. Embed and transform suspicious chunks.
7. Query Chroma for nearest source chunks.
8. Aggregate chunk hits into a source-document ranking.
9. Evaluate Hit@1, Hit@5, and Hit@10.
10. Save metrics, per-document retrieval outputs, memory accounting, plots, manifests, and a Markdown result summary.

The default chunking is `chunk_size=1200` characters with `overlap=200`. PAN plagiarism is often partial, so chunking preserves localized similarity that full-document embeddings can blur away. The default aggregation is `sum_top_n`, which sums the best few chunk similarities for each source document; this rewards repeated evidence without letting one noisy chunk dominate everything.

## Directory Structure

```text
.
├── pan11_retrieval/
│   ├── pan11.py              # PAN11 document loading and XML ground truth parsing
│   ├── chunking.py           # Configurable document-to-chunks mapping
│   ├── embeddings.py         # SentenceTransformers embedding wrapper
│   ├── compression.py        # Full, PCA, truncation, and int8 transforms
│   ├── chroma_store.py       # Persistent Chroma indexing and querying
│   ├── evaluation.py         # Aggregation and Hit@k computation
│   ├── memory.py             # Vector-size and compression accounting
│   ├── plotting.py           # Matplotlib PNG plots and result summary
│   ├── config.py             # Experiment dataclasses and reproducibility
│   └── experiment.py         # End-to-end orchestration
├── run_baseline.py
├── run_compression_experiments.py
├── plot_results.py
├── requirements.txt
└── PITFALLS_AND_THREATS.md
```

## Setup

Use a normal Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Run a small debug subset first:

```powershell
python run_compression_experiments.py --data-root C:\path\to\pan11 --debug-limit 20 --debug-source-limit 200
```

`--debug-limit` limits suspicious documents. `--debug-source-limit` limits source documents and is useful for fast smoke tests only; full evaluation should omit it.
Do not use debug limits for final analysis. Use `--sample-fraction 0.1` for a smaller but defensible subset experiment.

Run the full compression experiment:

```powershell
python run_compression_experiments.py --data-root C:\path\to\pan11
```

Run a seeded 10% subset experiment:

```powershell
python run_compression_experiments.py --data-root C:\path\to\pan11 --sample-fraction 0.1
```

The sampling mode randomly samples suspicious documents, keeps all true source documents required by those suspicious documents, and adds random source distractors until the source pool is about the same fraction of the full source pool. This is the recommended mode for a meaningful smaller experiment.

Run only the full-embedding baseline:

```powershell
python run_baseline.py --data-root C:\path\to\pan11
```

Regenerate plots and the Markdown summary from an existing metrics file:

```powershell
python plot_results.py --metrics-csv outputs\compression\tables\compression_comparison.csv --output-dir outputs\compression
```

## Outputs

The compression run writes:

```text
outputs/compression/
├── tables/
│   ├── baseline_results.csv
│   ├── compression_comparison.csv
│   ├── memory_vs_hit.csv
│   ├── latency_vs_hit.csv
│   └── vector_size_comparison.csv
├── per_document/
│   └── *_retrieval_results.csv
├── plots/
│   ├── hit_at_1_vs_dimension.png
│   ├── hit_at_5_vs_dimension.png
│   ├── hit_at_10_vs_dimension.png
│   ├── hit_vs_memory_per_vector.png
│   ├── hit_vs_total_memory.png
│   ├── compression_ratio_vs_hit.png
│   ├── method_hit_bar.png
│   ├── vector_memory_usage_bar.png
│   └── latency_vs_hit.png
├── manifests/
│   ├── config.json
│   └── *.json
└── RESULTS.md
```

## Compression Methods

Full baseline stores the original normalized float32 embeddings. PCA fits scikit-learn PCA on source chunk embeddings only, then transforms both source and suspicious chunks. Truncation keeps the first `k` dimensions directly. Scalar int8 quantization fits per-dimension symmetric scales on source embeddings, stores/accounting uses int8, and vectors are dequantized before insertion/querying because Chroma expects float embeddings. This keeps Chroma as the retrieval database while making the vector memory comparison explicit and reproducible.

## Metrics

Hit@k is the right baseline metric for this source-retrieval task because the retrieval question is whether any true source document appears in the top-k ranked candidates. The implementation supports multiple true sources per suspicious document.

Every condition records:

- vector dimensionality
- storage dtype
- bytes per scalar
- bytes per vector
- number of stored source chunk vectors
- estimated total source-vector memory
- compression ratio relative to the full float32 baseline
- indexing time
- query time
- Hit@1, Hit@5, Hit@10

Vector memory is central here because dimensionality reduction and quantization are useful only if their retrieval loss is justified by concrete savings in stored source vectors.
