| Configuration | hit@1 | hit@5 | hit@10 | MRR | dMRR vs baseline (95% paired CI) | better/worse | rerank s/q |
|---|---|---|---|---|---|---|---|
| Baseline: dense, bge-small, 1000/150 | 0.388 | 0.694 | 0.857 | 0.510 | - | - | 0 |
| Dense, bge-small-en-v1.5, 500/75 | 0.531 | 0.694 | 0.776 | 0.601 | +0.091 [-0.018, +0.202] | 15/12 | 0 |
| Dense, bge-small-en-v1.5, 2000/300 | 0.347 | 0.714 | 0.837 | 0.502 | -0.008 [-0.112, +0.089] | 17/14 | 0 |
| Dense, bge-small-en-v1.5, 1000/0 | 0.469 | 0.735 | 0.857 | 0.587 | +0.077 [-0.036, +0.188] | 21/10 | 0 |
| Dense, bge-small-en-v1.5, 1000/300 | 0.388 | 0.714 | 0.837 | 0.517 | +0.007 [-0.070, +0.082] | 12/15 | 0 |
| Dense, bge-base-en-v1.5, 1000/150 | 0.449 | 0.755 | 0.857 | 0.564 | +0.053 [-0.009, +0.127] | 11/9 | 0 |
| Dense, all-MiniLM-L6-v2, 1000/150 | 0.408 | 0.735 | 0.857 | 0.540 | +0.030 [-0.054, +0.115] | 13/11 | 0 |
| BM25 only, 1000/150 | 0.653 | 0.878 | 0.918 | 0.753 | +0.243 [+0.120, +0.366] * | 22/8 | 0 |
| Hybrid BM25 + dense (RRF), 1000/150 | 0.612 | 0.857 | 0.898 | 0.696 | +0.186 [+0.094, +0.287] * | 19/6 | 0 |
| Hybrid + rerank ms-marco-MiniLM-L-6-v2 | 0.510 | 0.837 | 0.857 | 0.628 | +0.117 [+0.026, +0.216] * | 17/11 | 0.087 |
| Hybrid + rerank bge-reranker-base | 0.673 | 0.837 | 0.898 | 0.745 | +0.235 [+0.116, +0.350] * | 25/5 | 0.521 |

`*` = paired 95% interval excludes 0. n=49 answerable-type questions; 10 comparisons against the baseline, so one starred row can appear by chance.
