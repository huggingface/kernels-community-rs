#!/usr/bin/env bash
cd "$(dirname "$0")/.." && nix build -L .#backendBundle.cuda && { [ -e .bench/baseline ] || uv run benchmarks/benchmark.py --freeze .bench/baseline; } && uv run benchmarks/benchmark.py --suite full --baseline .bench/baseline --output ".bench/full-$(date +%Y%m%d-%H%M%S).json" "$@"
