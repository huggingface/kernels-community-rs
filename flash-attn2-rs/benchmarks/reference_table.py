"""Render the reference suite's graph timings without importing Torch."""

import argparse
import json
from pathlib import Path

DTYPES = {"float16": "f16", "bfloat16": "bf16"}


def render(report):
    columns, operations, cells = [], [], {}
    for case in report["cases"]:
        name = case["name"]
        if not name.startswith("ref-"):
            raise ValueError(f"not a reference matrix case: {name}")
        operation, dtype, length = name[4:].rsplit("-", 2)
        column = f"{DTYPES[dtype]}, S={length}"
        if column not in columns:
            columns.append(column)
        if operation not in operations:
            operations.append(operation)
        cells[operation, column] = case["measurements"]["graph"]
    reference = report.get("reference")
    title = (
        "Rust speed relative to C++ per op"
        if reference
        else "Candidate speed relative to baseline per op"
    )
    lines = [
        title,
        "",
        "| Op | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for operation in operations:
        values = []
        for column in columns:
            measurement = cells.get((operation, column))
            if measurement is None:
                values.append("N/A")
            else:
                values.append(
                    f"{measurement['baseline_us'] / measurement['candidate_us']:.2f}x"
                )
        lines.append(f"| {operation} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Relative speed = C++ median time / Rust median time; above 1.00x means Rust is faster."
        if reference
        else "Relative speed = baseline median time / candidate median time; above 1.00x means the candidate is faster.",
        "CUDA events measure graph replay per operation, excluding Python overhead. Backward excludes forward setup.",
        "Shapes: B=2, Hq=8, D=64; Q=[B,S,Hq,D], K/V=[B,S,Hkv,D]. Hkv=8 except GQA, decode, and paged attention (Hkv=2).",
        "Decode uses Sq=1, Sk=S. Variable-length and paged cases use lengths [S,S−1]; paged K/V use 256-token pages.",
        "Causal attention is enabled for fwd_causal, fwd_gqa, varlen_fwd, varlen_bwd, and paged_fwd. Dropout is zero.",
        "Only f16/bf16 are benchmarked; these FlashAttention kernels do not support f32 inputs.",
        "",
    ]
    if reference:
        lines += [
            f"Reference: `{reference['repo_id']}` v{reference['version']}, commit `{reference['commit']}`.",
            "",
        ]
    if report["cases"]:
        worker = report["cases"][0]["workers"][0]
        lines += [
            f"GPU: {worker['gpu']}; Torch: {worker['torch']}; CUDA: {worker['cuda']}.",
            "",
        ]
    lines += [
        "Median kernel time (µs):",
        "",
        "| Op | Configuration | C++ (µs) | Rust (µs) | Relative speed |"
        if reference
        else "| Op | Configuration | Baseline (µs) | Candidate (µs) | Relative speed |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (operation, column), m in cells.items():
        lines.append(
            f"| {operation} | {column} | {m['baseline_us']:.2f} | {m['candidate_us']:.2f} | {m['baseline_us'] / m['candidate_us']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.report.read_text())), end="")
