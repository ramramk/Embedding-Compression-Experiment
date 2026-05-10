# PAN11 Retrieval Experiment Results

This experiment evaluates source-document retrieval with chunked suspicious/source documents, Chroma vector search, and document-level aggregation of chunk evidence.

## Best Hit@10

- Condition: pca_384
- Hit@10: 0.7692
- Bytes per vector: 1536

## Smallest Vector Representation

- Condition: pca_64
- Bytes per vector: 256
- Hit@10: 0.7308

## Full Metrics

| method     | condition      |    hit@1 |    hit@5 |   hit@10 |   num_evaluated_suspicious_documents |   embedding_dim | storage_dtype   |   bytes_per_scalar |   bytes_per_vector |   num_source_vectors |   estimated_total_vector_bytes |   compression_ratio_vs_full |   indexing_time_seconds |   query_time_seconds |   num_source_documents |   num_source_chunks |   num_suspicious_documents |
|:-----------|:---------------|---------:|---------:|---------:|-------------------------------------:|----------------:|:----------------|-------------------:|-------------------:|---------------------:|-------------------------------:|----------------------------:|------------------------:|---------------------:|-----------------------:|--------------------:|---------------------------:|
| full       | full           | 0.596154 | 0.634615 | 0.692308 |                                   52 |             384 | float32         |                  4 |               1536 |                27365 |                       42032640 |                         1   |                 45.551  |             13.0334  |                    116 |               27365 |                         52 |
| pca        | pca_384        | 0.576923 | 0.673077 | 0.769231 |                                   52 |             384 | float32         |                  4 |               1536 |                27365 |                       42032640 |                         1   |                 71.9281 |             10.5862  |                    116 |               27365 |                         52 |
| pca        | pca_256        | 0.576923 | 0.673077 | 0.730769 |                                   52 |             256 | float32         |                  4 |               1024 |                27365 |                       28021760 |                         1.5 |                 76.4943 |             14.2215  |                    116 |               27365 |                         52 |
| pca        | pca_128        | 0.557692 | 0.634615 | 0.75     |                                   52 |             128 | float32         |                  4 |                512 |                27365 |                       14010880 |                         3   |                 66.6554 |             11.108   |                    116 |               27365 |                         52 |
| pca        | pca_64         | 0.480769 | 0.615385 | 0.730769 |                                   52 |              64 | float32         |                  4 |                256 |                27365 |                        7005440 |                         6   |                 79.4128 |              8.88015 |                    116 |               27365 |                         52 |
| truncation | truncation_384 | 0.596154 | 0.634615 | 0.692308 |                                   52 |             384 | float32         |                  4 |               1536 |                27365 |                       42032640 |                         1   |                  0      |             13.0334  |                    116 |               27365 |                         52 |
| truncation | truncation_256 | 0.596154 | 0.634615 | 0.653846 |                                   52 |             256 | float32         |                  4 |               1024 |                27365 |                       28021760 |                         1.5 |                 73.7015 |             11.1859  |                    116 |               27365 |                         52 |
| truncation | truncation_128 | 0.5      | 0.596154 | 0.673077 |                                   52 |             128 | float32         |                  4 |                512 |                27365 |                       14010880 |                         3   |                 79.3681 |              8.4515  |                    116 |               27365 |                         52 |
| truncation | truncation_64  | 0.480769 | 0.576923 | 0.634615 |                                   52 |              64 | float32         |                  4 |                256 |                27365 |                        7005440 |                         6   |                 75.9814 |              9.36477 |                    116 |               27365 |                         52 |
| int8       | int8_full_dim  | 0.576923 | 0.634615 | 0.692308 |                                   52 |             384 | int8            |                  1 |                384 |                27365 |                       10508160 |                         4   |                136.753  |             12.8926  |                    116 |               27365 |                         52 |
