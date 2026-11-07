#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nix build -L .#backendBundle.cuda
output=".bench/reference-$(date -u +%Y%m%d-%H%M%S)"
uv run benchmarks/benchmark.py --reference-version 4 --suite reference --mode graph \
    --output "${output}.json" --markdown "${output}.md" "$@"
