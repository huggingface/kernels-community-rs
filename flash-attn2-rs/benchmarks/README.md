# Benchmarks

Compare the local Rust build with `kernels-community/flash-attn2` version 4:

```bash
bash benchmarks/run_reference.sh
```

This builds the CUDA bundle and writes timestamped JSON and Markdown reports to
`.bench/reference-*.{json,md}`. The reference version is required: the runner
resolves v4 to a Hub commit, records that commit and hashes both artifacts, and
never falls back to another version. Workers load each implementation in a
separate process using the same Python environment.

To use an existing build without rebuilding:

```bash
uv run benchmarks/benchmark.py \
  --candidate result --reference-version 4 --suite reference --mode graph \
  --output .bench/reference-v4.json --markdown .bench/reference-v4.md
```

The first table has operations as rows and f16/bf16 at sequence lengths 1024/4096
as columns. It reports **C++ median time / Rust median time** as relative speed,
formatted as `1.23x`: above `1.00x` means Rust is faster. A second table shows
C++ and Rust median times in microseconds alongside the same relative speed. Confidence intervals and measurement status are kept in JSON,
along with raw samples, correctness results, GPU activity, and environment metadata.

Timing uses CUDA events around repeated CUDA graph replay, excluding Python
validation and allocation overhead. GPU preprocessing is included; backward
forward-setup is excluded. Inputs are resident and shared between implementations.
The runner checks results before timing, validates graph replay, alternates
measurement order, and records interference from other GPU processes.

Cases cover dense, causal and grouped-query forward, dense backward,
variable-length forward/backward, KV-cache decode, and paged forward. B=2, Hq=8,
D=64, dropout=0. Hkv=8 except GQA/decode/paged (Hkv=2). Decode uses one query token;
other cases use Sq=Sk=S, with packed/paged lengths [S,S−1] and 256-token pages.
Causal attention is enabled for causal/GQA, variable-length, and paged rows.
Float32 is omitted because these kernels support f16/bf16 inputs only.

Select individual cases with `--cases` (see `--list`), or add
`--fail-on-regression` to make regressions and unreliable measurements fail the
command. Regenerating Markdown from saved reference-suite JSON needs only Python:

```bash
python3 benchmarks/reference_table.py .bench/reference-v4.json > .bench/reference-v4.md
```

Use `--baseline .bench/baseline` with `quick` or `full` to compare local Rust
artifacts. The `full` suite also includes noncausal packed D64 backward at 1024
and 4096 tokens in both f16 and bf16; these cases use unequal packed lengths.
