# Estimated API cost for this workload (not paid; local llama3.1:8b was used)
Measured on the hybrid run: 49 queries, 983 input / 90 output tokens per query (Llama tokenizer).
Prices: Anthropic list prices per MTok from platform.claude.com pricing page, checked 2026-10-26.
Claude 4.7+ tokenizers produce ~30% more tokens; the last column applies that.

| Model | $/query | $/1,000 queries | with +30% tokens |
|---|---|---|---|
| Claude Haiku 5.5 ($0.10 / $0.50) | 0.00014 | 0.14 | 0.19 |
| Claude Sonnet 5.5 ($2 / $10) | 0.00286 | 2.86 | 3.72 |
| Claude Opus 5.5 ($4 / $20) | 0.00573 | 5.73 | 7.45 |
| Claude Fable 5.1 ($10 / $50) | 0.01432 | 14.32 | 18.62 |

Local latency (16 GB M-series Mac, warm): retrieval ~50 ms, generation median 8.7 s (p95 11.6 s), cold start +4.3 s.
