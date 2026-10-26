#!/bin/bash
set -eo pipefail
export PYTHONPATH=src
rm -f data/processed/chunks_500_75.jsonl data/processed/chunks_2000_300.jsonl data/processed/chunks_1000_0.jsonl data/processed/chunks_1000_300.jsonl
rm -f eval/results/experiments.jsonl
echo "== chunk sweep (bge-small)"
python eval/experiments.py 2>&1 | grep -E "^ +[0-9]+ +[0-9]+|size|Error|Traceback"
for m in BAAI/bge-base-en-v1.5 sentence-transformers/all-MiniLM-L6-v2; do
  echo "== $m"
  python eval/experiments.py build 1000 150 $m > /tmp/build.log 2>&1 || { tail -5 /tmp/build.log; exit 1; }
  python eval/experiments.py eval 1000 150 $m > /tmp/eval.log 2>&1 || { tail -5 /tmp/eval.log; exit 1; }
  echo done
done
echo "== hybrid"
python eval/hybrid_eval.py 2>&1 | grep -E "^(dense|bm25|hybrid|method|still)|Error|Traceback"
echo "== rerank"
python eval/rerank_eval.py 2>&1 | grep -E "^(hybrid|cross|BAAI|method)|Error|Traceback"
python eval/build_results_table.py > /dev/null
echo "all done"
