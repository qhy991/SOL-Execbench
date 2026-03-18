#!/bin/bash

echo "Downloading SOL-ExecBench dataset..."
python scripts/download_solexecbench.py
echo "Done"
echo ""

echo "Downloading flashinfer-ai/flashinfer-trace dataset..."
hf download flashinfer-ai/flashinfer-trace --repo-type=dataset --revision 1.0 --local-dir data/flashinfer-trace
echo "Done"
