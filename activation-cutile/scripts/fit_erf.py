"""Fit the f32 `erf` used by the GELU kernels and check its accuracy.

cuTile has no `erf`, so `src/device.rs` approximates it. For a = |x|:

- a < S:  erf(x) = x + x * p(x^2 - D)
- a >= S: erf(x) = copysign(1 - 2^-|r|, x) with r = x + x * q(a - C), i.e.
  |r| approximates -log2(erfc(a)). q is fitted up to END; f32 erf is 1 from
  ~3.92, and beyond END q keeps growing, so 2^-|r| stays below half an ulp
  of 1 (checked for all f32, as are inf and NaN).

p and q have the same degree, so the kernel evaluates a single Horner chain
with per-element coefficients, plus one `exp2` (a single hardware
instruction, unlike `exp`). `x^2 - D` is a single `fma`.

The coefficients are near-minimax fits (Lawson's algorithm) in f64 against
Python's `math.erf`/`math.erfc`: p minimizes erf's relative error and q its
absolute error. They are then rounded to f32 one at a time, refitting the
remaining ones. S, C and D were picked (`--search`) to minimize the
worst-case error of the f32 evaluation.

The f32 evaluation is emulated in f64 with the rounding of every operation
(`fma` stays fused), and checked against `torch.erf` in f64 for all
non-negative f32 (the algorithm is exactly odd). `exp2` is f32 `torch.exp2` on CUDA (max error ~2 ulp), which
cuTile's `exp2` matches bit for bit on all f32 in [-28, -0.5].

Usage (needs numpy and torch with CUDA, e.g. in the kernel dev shell):

    python scripts/fit_erf.py           # fit with S, C and D below, check
    python scripts/fit_erf.py --search  # search S, C and D
"""

import math
import sys

import numpy as np
import torch

DEGREE = 6
S = 1.125
C = 2.0
D = 0.5
# End of the large-|x| fit; f32 erf is 1 from here on.
END = 4.0


def f32(v: float) -> float:
    return float(np.float32(v))


def lawson(basis, target, weight, iters=400):
    """Near-minimax: minimize max |weight * (basis @ c - target)|."""
    lw = np.full(len(target), 1.0 / len(target))
    for _ in range(iters):
        w = np.sqrt(lw) * weight
        c, *_ = np.linalg.lstsq(basis * w[:, None], target * w, rcond=None)
        lw = lw * (np.abs(weight * (basis @ c - target)) + 1e-300)
        lw /= lw.sum()
    return c


def fit_rounded(var, target, weight):
    """Fit, then round the coefficients to f32 from the constant term up."""
    fixed = []
    for k in range(DEGREE + 1):
        basis = np.stack([var**j for j in range(k, DEGREE + 1)], axis=1)
        partial = sum(c * var**j for j, c in enumerate(fixed)) if fixed else 0.0
        fixed.append(f32(lawson(basis, target - partial, weight)[0]))
    return fixed


def chebyshev(a, b, n=4000):
    k = np.arange(n)
    return (a + b) / 2 + (b - a) / 2 * np.cos(np.pi * (k + 0.5) / n)


def fit(s, c, d):
    # Small: erf(x) / x - 1 = p(x^2 - d), relative error.
    x = chebyshev(0.0, s)
    g = np.array([math.erf(v) / v for v in x])
    p = fit_rounded(x * x - d, g - 1.0, 1.0 / g)
    # Large: -log2(erfc(t)) / t - 1 = q(t - c); erf's error is
    # ln(2) * erfc(t) * t * (error of q).
    t = chebyshev(s, END)
    erfc = np.array([math.erfc(v) for v in t])
    q = fit_rounded(t - c, -np.log2(erfc) / t - 1.0, math.log(2) * erfc * t)
    return p, q


def rnd(v):
    return v.to(torch.float32).to(torch.float64)


def erf_f32(x, s, c, d, p, q):
    """The kernel's algorithm, on f32 values held in f64 tensors."""
    a = x.abs()
    big = a >= s
    z = torch.where(big, rnd(a - c), rnd(x * x - d))
    acc = torch.where(big, q[-1], p[-1]).to(x)
    for cp, cq in zip(reversed(p[:-1]), reversed(q[:-1])):
        acc = rnd(acc * z + torch.where(big, cq, cp))
    # f32 overflows to inf for huge |x|, like the kernel.
    r = (acc * x + x).float()
    tail = rnd(1.0 - torch.exp2(-r.abs()).double())
    tail = torch.where(x < 0, -tail, tail)
    return torch.where(big, tail, r.double())


def check(s, c, d, p, q, exhaustive=True):
    """Max error in ulps and the share of correctly rounded results."""
    worst, n, correct = 0.0, 0, 0
    if exhaustive:
        # All non-negative f32 up to and including inf.
        top = int(np.float32(np.inf).view(np.int32)) + 1
        chunks = [(i, min(i + (1 << 25), top)) for i in range(0, top, 1 << 25)]
    else:
        chunks = [None]
    for chunk in chunks:
        if chunk is None:
            x32 = torch.linspace(0, 8, 1 << 23, device="cuda")
        else:
            x32 = torch.arange(*chunk, device="cuda", dtype=torch.int32).view(torch.float32)
        x = x32.to(torch.float64)
        exact = torch.erf(x)
        cr = rnd(exact)
        ulp = torch.nextafter(cr.float(), torch.tensor(float("inf"), device="cuda")).double() - cr
        got = erf_f32(x, s, c, d, p, q)
        worst = max(worst, ((got - exact).abs() / ulp).max().item())
        correct += (got == cr).sum().item()
        n += x.numel()
    nan = erf_f32(torch.tensor([float("nan")], device="cuda", dtype=torch.float64), s, c, d, p, q)
    assert nan.isnan().all(), "erf(nan) must be nan"
    return worst, 100 * correct / n


def search():
    results = []
    for s in np.arange(1.0, 1.2001, 0.025):
        for c in [1.5, 2.0, 2.5]:
            for d in [0.0, 0.25, 0.5, 0.75]:
                p, q = fit(f32(s), c, d)
                results.append((check(f32(s), c, d, p, q, exhaustive=False), f32(s), c, d))
    results.sort()
    for _, s, c, d in results[:5]:
        p, q = fit(s, c, d)
        worst, cr = check(s, c, d, p, q)
        print(f"S = {s:.3f}, C = {c}, D = {d}: max {worst:.3f} ulp, {cr:.2f}% correctly rounded")


def main():
    assert math.erfc(END) < 2.0**-25, "f32 erf must be 1 from END on"
    if "--search" in sys.argv:
        search()
        return
    s = f32(S)
    p, q = fit(s, C, D)
    worst, cr = check(s, C, D, p, q)
    print(f"S = {S}, C = {C}, D = {D}: max {worst:.3f} ulp, {cr:.3f}% correctly rounded (all f32)")
    # Constant term first, as Rust literals.
    for name, coeffs in [("p", p), ("q", q)]:
        print(f"{name}:", ", ".join(str(np.float32(v)) + "f32" for v in coeffs))


if __name__ == "__main__":
    main()
