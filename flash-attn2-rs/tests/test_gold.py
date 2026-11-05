import itertools
import math

import kernels
import pytest
import torch
from torch.utils.checkpoint import checkpoint

flash_attn2_rs = kernels.get_kernel("kernels-community-rs/flash-attn2-rs", version=1)
flash_attn2 = kernels.get_kernel("kernels-community/flash-attn2", version=4)

# Generator-backed dropout needs `torch.Generator.philox_state` (Torch 2.14).
requires_philox_state = pytest.mark.skipif(
    not hasattr(torch.Generator, "philox_state"),
    reason="generator-backed dropout needs Torch 2.14",
)

KERNELS = [
    pytest.param(flash_attn2_rs, id="flash-attn2-rs"),
    pytest.param(flash_attn2, id="flash-attn2"),
]


def fwd_explicit(
    q,
    k,
    v,
    *,
    alibi_slopes=None,
    softmax_scale=None,
    causal=False,
    window_left=-1,
    window_right=-1,
    softcap=0.0,
):
    """Call the positional tvm-ffi forward op with keyword arguments."""
    return flash_attn2_rs.fwd_explicit(
        q,
        k,
        v,
        alibi_slopes,
        softmax_scale,
        causal,
        window_left,
        window_right,
        softcap,
    )


def reference_forward(
    q,
    k,
    v,
    *,
    alibi_slopes=None,
    softmax_scale=None,
    causal=False,
    window_left=-1,
    window_right=-1,
    softcap=0.0,
):
    """Run the reference forward and return its output and LSE."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    out, lse, _ = flash_attn2.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(window_left, window_right),
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_attn_probs=True,
    )
    return out, lse


def assert_all_close(actual, expected, *, atol, rtol):
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)


def test_dropout_replays_reference_rng():
    """Seed/offset pairs taken from the reference reproduce its dropout masks."""
    torch.manual_seed(0)
    # All cases share one generator, so the Philox offset advances across them.
    generator = torch.Generator(device="cuda").manual_seed(2**63 + 17)
    generator.set_offset(2**32 - 32)
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (24, 64, 96, 128, 160, 256):
            causal = head_dim % 64 == 0
            seqlen_q, seqlen_k = (97, 65) if causal else (65, 97)
            q = torch.randn(2, seqlen_q, 4, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(2, seqlen_k, 2, head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            expected = flash_attn2.fwd(
                q, k, v, p_dropout=0.17, is_causal=causal, gen=generator
            )
            seed, offset = expected[3].cpu().tolist()
            out, lse = flash_attn2_rs.fwd_explicit(
                q, k, v, None, None, causal, -1, -1, 0.0, 0.17, seed, offset
            )
            replay_out, replay_lse = flash_attn2_rs.fwd_explicit(
                q, k, v, None, None, causal, -1, -1, 0.0, 0.17, expected[3], 0
            )
            assert torch.equal(out, replay_out)
            assert torch.equal(lse, replay_lse)
            torch.testing.assert_close(
                out,
                expected[0],
                atol=3e-3 if dtype == torch.float16 else 3e-2,
                rtol=3e-2,
            )
            torch.testing.assert_close(lse, expected[1], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_layout(dtype):
    """Unaligned inputs, a caller-supplied output and empty keys."""
    torch.manual_seed(0)
    atol = 3e-3 if dtype == torch.float16 else 3e-2
    # Odd outer strides and a two-byte pointer offset require packing.
    q = torch.randn(2, 33, 4, 65, device="cuda", dtype=dtype)[..., 1:]
    k = torch.randn(2, 47, 2, 65, device="cuda", dtype=dtype)[..., 1:]
    expected = flash_attn2.fwd(q.contiguous(), k.contiguous(), k.contiguous())[0]
    storage = torch.full((q.numel() + 1,), 123, device="cuda", dtype=dtype)
    out_buffer = storage[1:].view(q.shape)
    out, _ = flash_attn2_rs.fwd_explicit(
        q, k, k, None, None, False, -1, -1, 0.0, out_buffer
    )
    assert out.data_ptr() == out_buffer.data_ptr()
    assert storage[0] == 123
    torch.testing.assert_close(out, expected, atol=atol, rtol=3e-2)

    # Empty keys must clear a strided output and produce +inf LSE.
    out_buffer = torch.empty_like(q, memory_format=torch.contiguous_format)
    out_buffer = out_buffer.transpose(0, 1)
    q = q.transpose(0, 1)
    empty = q[:, :0]
    out, lse = flash_attn2_rs.fwd_explicit(
        q, empty, empty, None, None, False, -1, -1, 0.0, out_buffer
    )
    assert out.data_ptr() == out_buffer.data_ptr()
    assert (out == 0).all()
    assert torch.isposinf(lse).all()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("module", KERNELS)
def test_forward_rejects_cpu_inputs(module):
    q = torch.zeros(1, 1, 1, 8, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        module.fwd(q, q, q)


@pytest.mark.kernels_ci
@requires_philox_state
def test_forward_dropout_generator():
    """Dropout reserves the same Philox state and mask as the reference."""
    torch.manual_seed(0)
    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
    generator = torch.Generator(device="cuda").manual_seed(17)
    reference_generator = torch.Generator(device="cuda").manual_seed(17)
    out, _, probabilities, rng_state = flash_attn2_rs.fwd(
        q, q, q, p_dropout=0.17, gen=generator, return_softmax=True
    )
    expected = flash_attn2.fwd(
        q, q, q, p_dropout=0.17, gen=reference_generator, return_softmax=True
    )
    assert torch.equal(rng_state, expected[3])
    assert generator.get_offset() == reference_generator.get_offset()
    assert probabilities.shape == expected[2].shape
    assert torch.equal(
        torch.signbit(probabilities[:, :, :33, :33]),
        torch.signbit(expected[2][:, :, :33, :33]),
    )
    torch.testing.assert_close(out, expected[0], atol=3e-3, rtol=3e-2)


@pytest.mark.kernels_ci
def test_forward_out_ignores_unused_generator():
    """A supplied output is returned and an unused generator is not touched."""
    torch.manual_seed(0)
    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    generator = torch.Generator(device="cpu").manual_seed(7)
    before = generator.get_state()
    # The reference casts numeric tensor arguments to integers.
    window = torch.tensor(1.5)
    result = flash_attn2_rs.fwd(
        q, q, q, out=out, gen=generator, window_size_left=window
    )
    assert result[0] is out
    torch.testing.assert_close(
        out,
        flash_attn2.fwd(q, q, q, window_size_left=window)[0],
        atol=3e-3,
        rtol=3e-2,
    )
    assert torch.equal(generator.get_state(), before)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("module", KERNELS)
@pytest.mark.parametrize(
    "options",
    [
        {"p_dropout": 1.0},
        {"p_dropout": 0.17, "is_causal": "false"},
        {"p_dropout": 0.17, "softmax_scale": "0.125"},
    ],
    ids=["dropout-one", "string-causal", "string-scale"],
)
def test_forward_rejects_invalid_scalars(module, options):
    """Invalid scalars raise without consuming the generator."""
    torch.manual_seed(0)
    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
    generator = torch.Generator(device="cuda").manual_seed(17)
    before = generator.get_state()
    with pytest.raises(RuntimeError):
        module.fwd(q, q, q, gen=generator, **options)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("module", KERNELS)
@pytest.mark.parametrize(
    ("extra", "first_error"),
    [({}, "softmax_scale"), ({"out": 7}, "out_")],
    ids=["scalar", "tensor"],
)
def test_forward_argument_conversion_order(module, extra, first_error):
    """Tensor conversion precedes scalar conversion, then native dtype checks."""
    torch.manual_seed(0)
    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(17)
    before = generator.get_state()
    with pytest.raises(RuntimeError, match=f"argument '{first_error}'"):
        module.fwd(q, q, q, gen=generator, p_dropout=0.17, softmax_scale="bad", **extra)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.kernels_ci
def test_forward_out_with_disjoint_rows():
    """Supplied outputs accept nonoverlapping rows outside the GCD proof."""
    torch.manual_seed(0)
    q = torch.randn(2, 2049, 2, 16, device="cuda", dtype=torch.float16)
    kv = q[:, :1]
    out = torch.empty(180344, device="cuda", dtype=q.dtype).as_strided(
        q.shape, (40, 88, 64, 1)
    )
    assert flash_attn2_rs.fwd(q, kv, kv, out=out)[0] is out
    torch.testing.assert_close(out, flash_attn2.fwd(q, kv, kv)[0], atol=3e-3, rtol=3e-2)


@pytest.mark.kernels_ci
def test_forward_batch_and_heads_grid():
    """Batch and heads occupy independent CUDA grid dimensions."""
    torch.manual_seed(0)
    q = torch.randn(257, 1, 256, 8, device="cuda", dtype=torch.float16)
    kv = torch.randn(257, 1, 1, 8, device="cuda", dtype=q.dtype)
    out = flash_attn2_rs.fwd(q, kv, kv)[0]
    torch.testing.assert_close(out, kv.expand_as(q), atol=0, rtol=0)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backward_strided_inputs(dtype):
    """Strided gradients, dropout replay and caller-supplied gradient buffers."""
    torch.manual_seed(0)
    atol, rtol = (0.008, 0.04) if dtype == torch.float16 else (0.06, 0.08)
    q = torch.empty_strided(
        (1, 33, 4, 128), (20000, 512, 128, 1), device="cuda", dtype=dtype
    ).normal_()
    k = torch.empty_strided(
        (1, 49, 2, 128), (16000, 256, 128, 1), device="cuda", dtype=dtype
    ).normal_()
    v, dout = torch.randn_like(k), torch.randn_like(q)
    for dropout_p, softcap in ((0.0, 0.0), (0.17, 0.0), (0.0, 2.0)):
        options = {"p_dropout": dropout_p, "softcap": softcap, "is_causal": True}
        out, lse, _, rng_state = flash_attn2_rs.fwd(q, k, v, **options)
        rng_before = torch.cuda.get_rng_state()
        args = (dout, q, k, v, out, lse)
        actual = flash_attn2_rs.bwd(
            *args, rng_state=rng_state, deterministic=True, **options
        )
        assert torch.equal(rng_before, torch.cuda.get_rng_state())
        expected = flash_attn2.bwd(
            *args, rng_state=rng_state, deterministic=True, **options
        )
        for x, y in zip(actual[:3], expected[:3]):
            assert x.stride() == y.stride()
            torch.testing.assert_close(x, y, atol=atol, rtol=rtol)

        buffers = [torch.empty_like(x) for x in (q, k, v)]
        # Native dQ destinations may be autograd leaves.
        buffers[0].requires_grad_()
        repeated = flash_attn2_rs.bwd(
            *args, *buffers, rng_state=rng_state, deterministic=True, **options
        )
        for x, y, buffer in zip(actual, repeated, buffers):
            assert y.data_ptr() == buffer.data_ptr()
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        torch.testing.assert_close(
            actual[3][..., :33], expected[3][..., :33], atol=2e-5, rtol=2e-5
        )


@pytest.mark.kernels_ci
@pytest.mark.parametrize("module", KERNELS)
@pytest.mark.parametrize(
    ("extra", "first_error"),
    [({"dq": 7, "rng_state": 7}, "dq_"), ({"rng_state": 7}, "softmax_scale")],
    ids=["gradient-buffer", "scalar"],
)
def test_backward_argument_conversion_order(module, extra, first_error):
    """Gradient buffers convert before scalars, and the saved RNG state after."""
    torch.manual_seed(0)
    q = torch.randn(1, 33, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 49, 2, 128, device="cuda", dtype=q.dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    out, lse, _, _ = flash_attn2_rs.fwd(q, k, v, is_causal=True)
    before = torch.cuda.get_rng_state()
    with pytest.raises(RuntimeError, match=f"argument '{first_error}'"):
        module.bwd(dout, q, k, v, out, lse, softmax_scale="bad", **extra)
    assert torch.equal(before, torch.cuda.get_rng_state())


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_k", "dropout_p", "causal", "head_dim"),
    [
        (33, 257, 0.0, False, 128),
        (65, 257, 0.17, True, 128),
        (513, 33, 0.17, False, 128),
        (513, 33, 0.0, False, 128),
        (257, 513, 0.0, False, 128),
        (257, 513, 0.17, False, 128),
        (257, 513, 0.17, True, 128),
        (513, 257, 0.17, True, 128),
        (128, 512, 0.17, True, 128),
        (256, 512, 0.17, False, 128),
        (511, 511, 0.17, True, 128),
        (512, 512, 0.17, True, 128),
        (400, 512, 0.17, True, 128),
        (512, 480, 0.17, True, 128),
        (768, 1000, 0.17, True, 128),
        (1023, 1024, 0.17, True, 128),
        (1024, 1024, 0.17, True, 128),
        (257, 513, 0.17, True, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_backward_shapes(seqlen_q, seqlen_k, dropout_p, causal, head_dim, dtype):
    """Direct dK/dV, key splitting, GQA, compact dQ and balanced causal."""
    torch.manual_seed(0)
    q = torch.randn(2, seqlen_q, 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, seqlen_k, 2, head_dim, device="cuda", dtype=q.dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    out, lse, _, rng_state = flash_attn2_rs.fwd(
        q, k, v, p_dropout=dropout_p, is_causal=causal
    )
    args = (dout, q, k, v, out, lse)
    options = {"p_dropout": dropout_p, "is_causal": causal, "rng_state": rng_state}
    actual = flash_attn2_rs.bwd(*args, **options)
    expected = flash_attn2.bwd(*args, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("shape", [(128, 128), (512, 512), (257, 513), (513, 257)])
def test_backward_fused_d32(dtype, kv_heads, causal, dropout_p, shape):
    torch.manual_seed(135)
    sq, sk = shape
    if sq == 257:
        q = torch.randn(2, 8, sq, 32, device="cuda", dtype=dtype).transpose(1, 2)
        k = torch.randn(2, kv_heads, sk, 32, device="cuda", dtype=dtype).transpose(1, 2)
    else:
        q = torch.randn(2, sq, 8, 32, device="cuda", dtype=dtype)
        k = torch.randn(2, sk, kv_heads, 32, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    options = dict(p_dropout=dropout_p, is_causal=causal)
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, **options)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, **options)
    expected = flash_attn2.bwd(*args, rng_state=rng, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [96, 192, 256])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize(
    "shape", [(128, 128), (512, 512), (257, 513), (513, 257), (1023, 1024)]
)
def test_backward_wide_heads(dtype, head_dim, kv_heads, causal, dropout_p, shape):
    """Wide and non-power-of-two heads, including padded tiles and tails."""
    if head_dim == 256 and dropout_p > 0:
        pytest.skip("reference has no D256 dropout backward below 144 KB shared memory")
    torch.manual_seed(136)
    sq, sk = shape
    q = torch.randn(2, sq, 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, sk, kv_heads, head_dim, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    options = dict(p_dropout=dropout_p, is_causal=causal)
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, **options)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, **options)
    expected = flash_attn2.bwd(*args, rng_state=rng, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.17, 0.5])
@pytest.mark.parametrize(
    "shape", [(128, 128), (512, 512), (257, 513), (513, 257), (1023, 1024)]
)
def test_backward_d256_dropout_oracle(dtype, kv_heads, causal, dropout_p, shape):
    """Below 144 KB of shared memory the reference launches nothing for this
    case and returns uninitialized gradients, so check against FP64 instead."""
    torch.manual_seed(137)
    sq, sk = shape
    q = torch.randn(2, sq, 8, 256, device="cuda", dtype=dtype)
    k = torch.randn(2, sk, kv_heads, 256, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    options = dict(p_dropout=dropout_p, is_causal=causal)
    out, lse, mask, rng = flash_attn2_rs.fwd(q, k, v, return_softmax=True, **options)
    actual = flash_attn2_rs.bwd(dout, q, k, v, out, lse, rng_state=rng, **options)[:3]

    group = 8 // kv_heads
    q64, k64, v64 = (t.double().requires_grad_() for t in (q, k, v))
    scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        q64,
        k64.repeat_interleave(group, dim=2),
    ) * 256**-0.5
    if causal:
        rows = torch.arange(sq, device="cuda")[:, None]
        cols = torch.arange(sk, device="cuda")[None, :]
        scores = scores.masked_fill(cols > rows + sk - sq, float("-inf"))
    # Rows with no visible key produce zero output and gradients.
    probs = scores.softmax(-1).nan_to_num()
    keep = (mask[:, :, :sq, :sk] >= 0).double() / (1 - dropout_p)
    expected_out = torch.einsum(
        "bhqk,bkhd->bqhd",
        probs * keep,
        v64.repeat_interleave(group, dim=2),
    )
    expected = torch.autograd.grad(expected_out, (q64, k64, v64), dout.double())
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(
        [t.float() for t in actual],
        [t.float() for t in expected],
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("dropout_p", [0.17, 0.5])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_split_dropout(dtype, kv_heads, dropout_p, causal):
    """Full split-query tiles preserve the dropout stream and all three gradients."""
    torch.manual_seed(134)
    q = torch.randn(2, 512, 8, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 512, kv_heads, 128, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, p_dropout=dropout_p, is_causal=causal)
    args = (dout, q, k, v, out, lse)
    options = dict(p_dropout=dropout_p, rng_state=rng, is_causal=causal)
    actual = flash_attn2_rs.bwd(*args, **options)
    expected = flash_attn2.bwd(*args, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [(-1, -1), (256, 0)])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("head_dim", [32, 128])
def test_backward_query_partitions_masked(dtype, window, dropout_p, batch_size, head_dim):
    torch.manual_seed(132)
    q = torch.randn(batch_size, 257, 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, 511, 2, head_dim, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    slopes = torch.linspace(0.01, 0.08, 8, device="cuda")
    options = dict(
        alibi_slopes=slopes,
        p_dropout=dropout_p,
        window_size_left=window[0],
        window_size_right=window[1],
    )
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, **options)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, **options)
    expected = flash_attn2.bwd(*args, rng_state=rng, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [511, 512])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_backward_query_partitions_mha(dtype, causal, seqlen, dropout_p):
    torch.manual_seed(133)
    q = torch.randn(1, seqlen, 8, 128, device="cuda", dtype=dtype)
    k, v, dout = (torch.randn_like(q) for _ in range(3))
    options = dict(p_dropout=dropout_p, is_causal=causal)
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, **options)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, **options)
    expected = flash_attn2.bwd(*args, rng_state=rng, deterministic=True, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize(
    "function",
    [
        "flash_attn_func",
        "flash_attn_qkvpacked_func",
        "flash_attn_kvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn_varlen_qkvpacked_func",
        "flash_attn_varlen_kvpacked_func",
    ],
)
def test_autograd(function):
    """Python wrappers match the reference with head padding and dropout."""
    torch.manual_seed(0)
    q = torch.randn(1, 17, 4, 63, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 23, 2, 63, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 17, 17], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 23, 23], device="cuda", dtype=torch.int32)
    q_varlen, k_varlen, v_varlen = q[0], k[0], v[0]
    inputs, metadata = {
        "flash_attn_func": ((q, k, v), ()),
        "flash_attn_qkvpacked_func": ((torch.stack((q, q, q), 2),), ()),
        "flash_attn_kvpacked_func": ((q, torch.stack((k, v), 2)), ()),
        "flash_attn_varlen_func": (
            (q_varlen, k_varlen, v_varlen),
            (cu_seqlens_q, cu_seqlens_k, 17, 23),
        ),
        "flash_attn_varlen_qkvpacked_func": (
            (torch.stack((q_varlen, q_varlen, q_varlen), 1),),
            (cu_seqlens_q, 17),
        ),
        "flash_attn_varlen_kvpacked_func": (
            (q_varlen, torch.stack((k_varlen, v_varlen), 1)),
            (cu_seqlens_q, cu_seqlens_k, 17, 23),
        ),
    }[function]

    actual_inputs = [x.detach().requires_grad_() for x in inputs]
    expected_inputs = [x.detach().requires_grad_() for x in inputs]
    rng_state = torch.cuda.get_rng_state()
    actual = getattr(flash_attn2_rs, function)(
        *actual_inputs, *metadata, dropout_p=0.17, causal=True
    )
    rng_after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_state)
    expected = getattr(flash_attn2, function)(
        *expected_inputs, *metadata, dropout_p=0.17, causal=True
    )
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.03)

    actual_grads = torch.autograd.grad(actual.sum(), actual_inputs)
    expected_grads = torch.autograd.grad(expected.sum(), expected_inputs)
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    assert_all_close(actual_grads, expected_grads, atol=0.008, rtol=0.04)


@requires_philox_state
@pytest.mark.parametrize("varlen", [False, True], ids=["dense", "varlen"])
def test_compiled_dropout(varlen):
    """Compiled dropout keeps identical calls independent and replayable."""
    torch.manual_seed(0)
    q = torch.randn(1, 5, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 7, 1, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    metadata = ()
    attention = flash_attn2_rs.flash_attn_func
    if varlen:
        q, k, v = q[0], k[0], v[0]
        cu_seqlens_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int32)
        metadata = (cu_seqlens_q, cu_seqlens_k, 3, 4)
        attention = flash_attn2_rs.flash_attn_varlen_func
    inputs = tuple(x.detach().requires_grad_() for x in (q, k, v))

    def twice(*xs):
        return tuple(
            attention(*xs, *metadata, dropout_p=0.17, deterministic=True)
            for _ in range(2)
        )

    def recomputed(*xs):
        return checkpoint(twice, *xs, use_reentrant=False, preserve_rng_state=True)

    # AOT must not merge identical random forwards or reuse their saved masks.
    rng_state = torch.cuda.get_rng_state()
    expected = twice(*inputs)
    rng_after = torch.cuda.get_rng_state()
    expected_grads = torch.autograd.grad(sum(x.sum() for x in expected), inputs)
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    torch.cuda.set_rng_state(rng_state)
    actual = torch.compile(twice, fullgraph=True)(*inputs)
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    actual_grads = torch.autograd.grad(sum(x.sum() for x in actual), inputs)
    assert_all_close(
        (*actual, *actual_grads), (*expected, *expected_grads), atol=0, rtol=0
    )
    assert torch.equal(rng_after, torch.cuda.get_rng_state())

    # fallback_random keeps eager RNG so masks compare bitwise.
    with torch._inductor.config.patch(fallback_random=True):
        torch.cuda.set_rng_state(rng_state)
        replay = torch.compile(recomputed, fullgraph=True)(*inputs)
        assert torch.equal(rng_after, torch.cuda.get_rng_state())
        torch.rand(11, device="cuda")
        rng_before_backward = torch.cuda.get_rng_state()
        replay_grads = torch.autograd.grad(sum(x.sum() for x in replay), inputs)
        assert torch.equal(rng_before_backward, torch.cuda.get_rng_state())
        assert_all_close(
            (*replay, *replay_grads), (*expected, *expected_grads), atol=0, rtol=0
        )


def varlen_inputs():
    """Three sequences, including an empty one, with GQA heads."""
    q = torch.randn(25, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(31, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.tensor([0, 17, 17, 25], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 19, 19, 31], device="cuda", dtype=torch.int32)
    return q, k, v, cu_seqlens_q, cu_seqlens_k


def paged_kv_cache(dtype):
    """Four 256-row pages and a block table for three sequences."""
    k_pages = torch.randn(4, 256, 2, 64, device="cuda", dtype=dtype)
    v_pages = torch.randn_like(k_pages)
    block_table = torch.tensor(
        [[2, 0], [1, 3], [3, 1]], device="cuda", dtype=torch.int32
    )
    return k_pages, v_pages, block_table


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "dropout_p", [0.0, pytest.param(0.17, marks=requires_philox_state)]
)
def test_varlen_forward_backward(dropout_p):
    """Packed forward and backward with GQA, empty sequences and dropout."""
    torch.manual_seed(0)
    q, k, v, cu_seqlens_q, cu_seqlens_k = varlen_inputs()
    options = {
        "max_seqlen_q": 17,
        "max_seqlen_k": 19,
        "p_dropout": dropout_p,
        "is_causal": True,
        "zero_tensors": True,
    }
    rng_state = torch.cuda.get_rng_state()
    actual = flash_attn2_rs.varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, **options)
    rng_after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_state)
    expected = flash_attn2.varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, **options)
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    assert_all_close(actual[:2], expected[:2], atol=0.005, rtol=0.03)
    if dropout_p:
        assert torch.equal(actual[3], expected[3])

    args = (torch.ones_like(q), q, k, v, actual[0], actual[1])
    args += (cu_seqlens_q, cu_seqlens_k)
    actual_grads = flash_attn2_rs.varlen_bwd(*args, rng_state=actual[3], **options)
    expected_grads = flash_attn2.varlen_bwd(*args, rng_state=actual[3], **options)
    assert actual_grads[3].shape == (4, 25 + 128 * 3)
    torch.testing.assert_close(actual_grads[3], expected_grads[3], atol=2e-4, rtol=2e-5)
    assert_all_close(actual_grads[:3], expected_grads[:3], atol=0.008, rtol=0.04)
    assert torch.equal(rng_after, torch.cuda.get_rng_state())


@pytest.mark.kernels_ci
def test_varlen_forward_empty_out():
    """Empty destinations have no overlapping addresses, even with zero strides."""
    torch.manual_seed(0)
    q, k, v, cu_seqlens_q, cu_seqlens_k = varlen_inputs()
    out = q[:0].as_strided((0, 4, 64), (0, 0, 1))
    result = flash_attn2_rs.varlen_fwd(
        q[:0],
        k,
        v,
        torch.zeros_like(cu_seqlens_q),
        cu_seqlens_k,
        out=out,
        max_seqlen_q=1,
        max_seqlen_k=19,
    )
    assert result[0] is out


@pytest.mark.kernels_ci
@requires_philox_state
def test_paged_varlen_forward_matches_packed():
    """Paged dropout matches packed keys, since the reference paged path ignores dropout."""
    torch.manual_seed(0)
    q, _, _, cu_seqlens_q, _ = varlen_inputs()
    k_pages, v_pages, block_table = paged_kv_cache(q.dtype)
    cu_seqlens_k = torch.tensor([0, 270, 270, 301], device="cuda", dtype=torch.int32)
    k_packed, v_packed = (
        torch.cat((pages[2], pages[0, :14], pages[3, :31]))
        for pages in (k_pages, v_pages)
    )
    options = {
        "max_seqlen_q": 17,
        "max_seqlen_k": 270,
        "p_dropout": 0.17,
        "is_causal": True,
    }
    rng_state = torch.cuda.get_rng_state()
    actual = flash_attn2_rs.varlen_fwd(
        q,
        k_pages,
        v_pages,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table=block_table,
        **options,
    )
    rng_after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_state)
    expected = flash_attn2.varlen_fwd(
        q, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, **options
    )
    assert torch.equal(rng_after, torch.cuda.get_rng_state())
    assert_all_close(actual[:2], expected[:2], atol=0.005, rtol=0.03)


@pytest.mark.kernels_ci
def test_paged_varlen_single_query_split():
    """Long single-query paged attention uses a native split/merge path."""
    torch.manual_seed(0)
    k_pages, v_pages, block_table = paged_kv_cache(torch.float16)
    q = torch.randn(3, 2, 64, device="cuda", dtype=torch.float16)
    cu_seqlens_q = torch.arange(4, device="cuda", dtype=torch.int32)
    args = (q, k_pages, v_pages, cu_seqlens_q, cu_seqlens_q * 512)
    options = {"block_table": block_table, "max_seqlen_q": 1, "max_seqlen_k": 512}
    actual = flash_attn2_rs.varlen_fwd(*args, **options)
    expected = flash_attn2.varlen_fwd(*args, **options)
    assert_all_close(actual[:2], expected[:2], atol=0.005, rtol=0.03)


@pytest.mark.kernels_ci
def test_varlen_grouped_decode_lse():
    """Grouped decode returns contiguous head-major LSE that matches FP32."""
    torch.manual_seed(0)
    q = torch.randn(16, 8, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(16, 1024, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    cu_seqlens_q = torch.arange(17, device="cuda", dtype=torch.int32)
    args = (q, k.flatten(0, 1), v.flatten(0, 1), cu_seqlens_q, cu_seqlens_q * 1024)
    options = {"max_seqlen_q": 1, "max_seqlen_k": 1024}
    actual = flash_attn2_rs.varlen_fwd(*args, **options)
    expected = flash_attn2.varlen_fwd(*args, **options)
    torch.testing.assert_close(actual[0], expected[0], atol=0.005, rtol=0.03)
    scores = torch.einsum(
        "bhd,bkhd->bhk", q.float(), k.float().repeat_interleave(4, 2)
    ) / math.sqrt(128)
    torch.testing.assert_close(actual[1], scores.logsumexp(-1).T, atol=0.005, rtol=0.03)
    assert actual[1].is_contiguous()


@pytest.mark.kernels_ci
def test_paged_varlen_split_partials():
    """Split partials use max_seqlen_q strides even for empty or short sequences."""
    torch.manual_seed(0)
    k_pages, v_pages, block_table = paged_kv_cache(torch.float16)
    q = torch.randn(82, 4, 64, device="cuda", dtype=torch.float16)
    cu_seqlens_q = torch.tensor([0, 65, 65, 82], device="cuda", dtype=torch.int32)
    cu_seqlens_k = torch.arange(4, device="cuda", dtype=torch.int32) * 512
    args = (q, k_pages, v_pages, cu_seqlens_q, cu_seqlens_k)
    options = {
        "block_table": block_table,
        "max_seqlen_q": 65,
        "max_seqlen_k": 512,
        "is_causal": True,
    }
    actual = flash_attn2_rs.varlen_fwd(*args, **options)
    expected = flash_attn2.varlen_fwd(*args, **options)
    assert_all_close(actual[:2], expected[:2], atol=0.005, rtol=0.03)


def kv_cache_with_grad(head_dim=16, capacity=4):
    q = torch.randn(1, 1, 1, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(1, capacity, 1, head_dim, device="cuda", dtype=q.dtype, requires_grad=True)
    v_cache = torch.randn_like(k_cache, requires_grad=True)
    return q, k_cache, v_cache


@pytest.mark.kernels_ci
@pytest.mark.parametrize("head_dim,capacity", [(16, 4), (128, 1025)])
def test_kvcache_rejected_append_preserves_cache(head_dim, capacity):
    """A rejected append keeps cache contents, versions and existing graphs."""
    torch.manual_seed(0)
    q, k_cache, v_cache = kv_cache_with_grad(head_dim, capacity)
    saved = (k_cache.detach().clone(), v_cache.detach().clone())
    versions = (k_cache._version, v_cache._version)
    loss = k_cache.square().float().sum() + v_cache.square().float().sum()
    with pytest.raises(RuntimeError):
        flash_attn2_rs.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=q, v=q.float(), cache_seqlens=1, num_splits=1
        )
    assert (k_cache._version, v_cache._version) == versions
    assert torch.equal(k_cache, saved[0])
    assert torch.equal(v_cache, saved[1])
    loss.backward()
    torch.testing.assert_close(k_cache.grad, 2 * saved[0])
    torch.testing.assert_close(v_cache.grad, 2 * saved[1])


@pytest.mark.kernels_ci
@pytest.mark.parametrize("head_dim,capacity", [(16, 4), (128, 1025)])
def test_kvcache_append_invalidates_saved_cache(head_dim, capacity):
    """Native cache writes bump versions, so stale saved values fail backward."""
    torch.manual_seed(0)
    q, k_cache, v_cache = kv_cache_with_grad(head_dim, capacity)
    versions = (k_cache._version, v_cache._version)
    loss = k_cache.square().float().sum() + v_cache.square().float().sum()
    flash_attn2_rs.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=q, v=q, cache_seqlens=1, num_splits=1
    )
    assert k_cache._version > versions[0]
    assert v_cache._version > versions[1]
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


def padded_kv_cache(head_dim=7):
    """A zeroed cache whose head size needs padding to a multiple of 8."""
    q = torch.ones(1, 1, 1, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.zeros(1, 4, 1, head_dim, device="cuda", dtype=q.dtype)
    cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
    return q, k_cache, cache_seqlens


@pytest.mark.kernels_ci
def test_kvcache_rejects_overlapping_padded_cache():
    """Head padding must not hide invalid original cache destinations."""
    q, k_cache, cache_seqlens = padded_kv_cache()
    for v_cache in (k_cache, torch.zeros_like(q).expand_as(k_cache)):
        with pytest.raises((ValueError, RuntimeError)):
            flash_attn2_rs.fwd_kvcache(
                q, k_cache, v_cache, k=q, v=q, seqlens_k=cache_seqlens
            )
        assert torch.count_nonzero(k_cache) == 0
        assert torch.count_nonzero(v_cache) == 0


@pytest.mark.kernels_ci
def test_kvcache_empty_append_shares_storage():
    """Empty appends do not write, so K and V may share storage."""
    q, k_cache, cache_seqlens = padded_kv_cache()
    empty = q[:, :0]
    args = (q, k_cache, k_cache)
    options = {"k": empty, "v": empty, "seqlens_k": cache_seqlens}
    actual = flash_attn2_rs.fwd_kvcache(*args, **options)
    expected = flash_attn2.fwd_kvcache(*args, **options)
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)


@pytest.mark.kernels_ci
def test_kvcache_rejects_seqlens_aliasing_cache():
    """Head padding must not hide metadata that shares writable cache storage."""
    q, k_cache, _ = padded_kv_cache()
    cache_seqlens = k_cache.flatten().view(torch.int32)[:1]
    cache_seqlens.fill_(1)
    saved = k_cache.clone()
    with pytest.raises((ValueError, RuntimeError), match="must not overlap"):
        flash_attn2_rs.fwd_kvcache(
            q, k_cache, torch.zeros_like(k_cache), k=q, v=q, seqlens_k=cache_seqlens
        )
    assert torch.equal(k_cache, saved)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("head_dim", [8, 7])
@pytest.mark.parametrize(
    ("argument", "dtype"),
    [("leftpad_k", torch.int32), ("alibi_slopes", torch.float32)],
    ids=["leftpad_k", "alibi_slopes"],
)
def test_kvcache_rejects_metadata_aliasing_cache(head_dim, argument, dtype):
    """Attention-only metadata must also survive the preceding cache append."""
    q, k_cache, cache_seqlens = padded_kv_cache(head_dim)
    v_cache = torch.zeros_like(k_cache)
    metadata = k_cache.flatten().view(dtype)[:1]
    versions = (k_cache._version, v_cache._version)
    with pytest.raises((ValueError, RuntimeError), match="must not overlap"):
        flash_attn2_rs.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            k=q,
            v=q,
            seqlens_k=cache_seqlens,
            **{argument: metadata},
        )
    assert torch.count_nonzero(k_cache) == 0
    assert torch.count_nonzero(v_cache) == 0
    assert (k_cache._version, v_cache._version) == versions


@pytest.mark.kernels_ci
def test_kvcache_interleaved_axes():
    """Three interleaved axes have disjoint rows despite overlapping spans."""
    torch.manual_seed(0)
    storage = torch.randn(2, 80, device="cuda", dtype=torch.float16)
    reference_storage = storage.clone()
    q = torch.randn(2, 1, 2, 8, device="cuda", dtype=storage.dtype)
    cache_seqlens = torch.ones(2, device="cuda", dtype=torch.int32)
    results = []
    for module, backing in (
        (flash_attn2_rs, storage),
        (flash_attn2, reference_storage),
    ):
        k_cache, v_cache = (
            row.as_strided((2, 2, 2, 8), (16, 24, 32, 1)) for row in backing
        )
        results.append(
            module.fwd_kvcache(q, k_cache, v_cache, k=q, v=q, seqlens_k=cache_seqlens)
        )
    assert_all_close(*results, atol=0.007, rtol=0.04)
    assert torch.equal(storage, reference_storage)


@pytest.mark.kernels_ci
def test_kvcache_padded_batch_strides():
    """Finite rows stay disjoint despite different padded batch strides."""
    rows, step, head_dim = 4097, 8192, 8
    batch_stride = (rows * (step + 1) + 135) // 8 * 8
    storage = torch.empty(
        batch_stride + 8 + (rows - 1) * (step + 1) + 2 * head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    k_cache = storage.as_strided(
        (2, rows, 1, head_dim), (batch_stride, step, head_dim, 1)
    )
    v_cache = storage.as_strided(
        (2, rows, 1, head_dim), (batch_stride + 8, step + 1, head_dim, 1), head_dim
    )
    q = torch.ones(2, 1, 1, head_dim, device="cuda", dtype=storage.dtype)
    v = (
        torch.arange(head_dim, device="cuda", dtype=storage.dtype)
        .reshape(1, 1, 1, head_dim)
        .expand_as(q)
        .contiguous()
    )
    cache_seqlens = torch.zeros(2, device="cuda", dtype=torch.int32)
    out = flash_attn2_rs.fwd_kvcache(
        q, k_cache, v_cache, k=q, v=v, seqlens_k=cache_seqlens
    )[0]
    torch.testing.assert_close(out, v)


@pytest.mark.kernels_ci
def test_kvcache_append_reads_same_allocation():
    """Old cache rows are read while appending elsewhere in the same allocation."""
    torch.manual_seed(0)
    k_cache = torch.randn(1, 8, 1, 16, device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
    q, k, v = k_cache[:, :2], k_cache[:, :1], v_cache[:, :1]
    cache_seqlens = torch.tensor([3], device="cuda", dtype=torch.int32)
    expected = flash_attn2.fwd_kvcache(
        q.clone(),
        k_cache_ref,
        v_cache_ref,
        k=k.clone(),
        v=v.clone(),
        seqlens_k=cache_seqlens,
    )
    actual = flash_attn2_rs.fwd_kvcache(
        q, k_cache, v_cache, k=k, v=v, seqlens_k=cache_seqlens
    )
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.kernels_ci
def test_kvcache_paged_rotary():
    """Split paged attention with rotary appends, full windows and empty appends."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 63, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 2, 63, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    k_cache = torch.randn(2, 256, 2, 63, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
    cache_seqlens = torch.tensor([255], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    angles = torch.randn(512, 32, device="cuda")
    options = {
        "k": k,
        "v": v,
        "rotary_cos": angles.cos().to(q.dtype),
        "rotary_sin": angles.sin().to(q.dtype),
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "causal": True,
        "num_splits": 3,
        "return_softmax_lse": True,
    }
    actual = flash_attn2_rs.flash_attn_with_kvcache(q, k_cache, v_cache, **options)
    expected = flash_attn2.flash_attn_with_kvcache(
        q, k_cache_ref, v_cache_ref, **options
    )
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    torch.testing.assert_close(k_cache, k_cache_ref, atol=0.004, rtol=0.004)
    assert torch.equal(v_cache, v_cache_ref)
    assert cache_seqlens.item() == 255

    # Full-capacity windows are unrestricted for noncausal query rotation.
    full_window = {**options, "causal": False, "window_size": (512, 512)}
    actual = flash_attn2_rs.flash_attn_with_kvcache(q, k_cache, v_cache, **full_window)
    expected = flash_attn2.flash_attn_with_kvcache(
        q, k_cache_ref, v_cache_ref, **full_window
    )
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)

    # The reference validates rotary tables but does not rotate Q for an empty append.
    options.update(k=k[:, :0], v=v[:, :0])
    saved = (k_cache.clone(), v_cache.clone())
    actual = flash_attn2_rs.flash_attn_with_kvcache(q, k_cache, v_cache, **options)
    expected = flash_attn2.flash_attn_with_kvcache(
        q, k_cache_ref, v_cache_ref, **options
    )
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    assert torch.equal(k_cache, saved[0])
    assert torch.equal(v_cache, saved[1])


@pytest.mark.kernels_ci
def test_kvcache_shared_storage_head_strides():
    """K/V views in shared storage can be disjoint with different head strides."""
    torch.manual_seed(0)
    storage = torch.randn(2, 8, 6, 64, device="cuda", dtype=torch.float16)
    reference_storage = storage.clone()
    q = torch.randn(1, 1, 4, 64, device="cuda", dtype=storage.dtype)
    k = torch.randn(1, 1, 2, 64, device="cuda", dtype=storage.dtype)
    v = torch.randn_like(k)
    cache_seqlens = torch.tensor([2], device="cuda", dtype=torch.int32)
    # The reference ignores sine when no cosine is supplied.
    ignored_sine = torch.tensor(0.0)
    results = []
    for module, backing in (
        (flash_attn2_rs, storage),
        (flash_attn2, reference_storage),
    ):
        results.append(
            module.fwd_kvcache(
                q,
                backing[:, :, :2],
                backing[:, :, 2::2],
                k,
                v,
                cache_seqlens,
                rotary_sin=ignored_sine,
            )
        )
    assert_all_close(*results, atol=0.007, rtol=0.04)
    assert torch.equal(storage, reference_storage)

    out, lse = flash_attn2_rs.fwd_kvcache(
        q,
        storage[:, :, :2],
        storage[:, :, 2::2],
        seqlens_k=torch.zeros_like(cache_seqlens),
        num_splits=128,
    )
    assert torch.count_nonzero(out) == 0
    assert torch.isposinf(lse).all()


@pytest.mark.kernels_ci
def test_kvcache_packed_disjoint_views():
    """Read-only sources and writable caches may occupy disjoint packed views."""
    torch.manual_seed(0)
    storage = torch.randn(1, 8, 3, 2, 64, device="cuda", dtype=torch.float16)
    reference_storage = storage.clone()
    cache_seqlens = torch.tensor([4], device="cuda", dtype=torch.int32)
    results = []
    for module, backing in (
        (flash_attn2_rs, storage),
        (flash_attn2, reference_storage),
    ):
        q, k, v = backing[:, :1, 0], backing[:, 1:2, 0], backing[:, 2:3, 0]
        k_cache, v_cache = backing[:, :, 1], backing[:, :, 2]
        results.append(module.fwd_kvcache(q, k_cache, v_cache, k, v, cache_seqlens))
    assert_all_close(*results, atol=0.007, rtol=0.04)
    assert torch.equal(storage, reference_storage)


@pytest.mark.kernels_ci
def test_kvcache_interleaved_batch_gaps():
    """Source batches may span adjacent gaps in physically interleaved caches."""
    torch.manual_seed(0)
    storage = torch.randn(256, device="cuda", dtype=torch.float16)
    reference_storage = storage.clone()
    cache_seqlens = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    results = []
    for module, backing in (
        (flash_attn2_rs, storage),
        (flash_attn2, reference_storage),
    ):
        q, k, v = (
            backing.as_strided((2, 1, 1, 8), (8, 8, 8, 1), offset)
            for offset in (16, 48, 80)
        )
        k_cache, v_cache = (
            backing.as_strided((2, 4, 1, 8), (32, 64, 8, 1), offset)
            for offset in (0, 8)
        )
        results.append(module.fwd_kvcache(q, k_cache, v_cache, k, v, cache_seqlens))
    assert_all_close(*results, atol=0.007, rtol=0.04)
    assert torch.equal(storage, reference_storage)


@pytest.mark.kernels_ci
def test_kvcache_grouped_decode_layout():
    """Grouped decoding keeps an interleaved query layout and fills `out` in head order."""
    torch.manual_seed(0)
    q = (
        torch.randn(8, 2, 128, device="cuda", dtype=torch.float16)
        .transpose(0, 1)
        .unsqueeze(1)
    )
    k_cache = torch.randn(2, 1024, 2, 128, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    actual = flash_attn2_rs.fwd_kvcache(q, k_cache, v_cache, num_splits=0)
    expected = flash_attn2.fwd_kvcache(q, k_cache, v_cache, num_splits=0)
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    assert actual[0].stride() == expected[0].stride()

    out = torch.empty((2, 4, 2, 128), device="cuda", dtype=q.dtype)
    expected_out = torch.empty_like(out)
    actual = flash_attn2_rs.fwd_kvcache(q, k_cache, v_cache, out=out, num_splits=0)
    expected = flash_attn2.fwd_kvcache(
        q, k_cache, v_cache, out=expected_out, num_splits=0
    )
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    torch.testing.assert_close(out, expected_out, atol=0.007, rtol=0.04)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("cache_len", [1000, 4096])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 192, 256])
def test_kvcache_fused_append_head_dims(dtype, kv_heads, cache_len, head_dim):
    """Single-token decode with an in-kernel append for every head-dim bucket."""
    torch.manual_seed(0)
    batch, heads = 2, 8
    q = torch.randn(batch, 1, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, 1, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    k_cache = torch.randn(
        batch, cache_len + 64, kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
    # Ragged lengths leave the last key tile partially filled.
    cache_seqlens = torch.tensor(
        [cache_len, cache_len - 37], device="cuda", dtype=torch.int32
    )
    options = {"k": k, "v": v, "seqlens_k": cache_seqlens}
    actual = flash_attn2_rs.fwd_kvcache(q, k_cache, v_cache, **options)
    expected = flash_attn2.fwd_kvcache(q, k_cache_ref, v_cache_ref, **options)
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("cache_len", [1000, 4096])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
def test_kvcache_unsplit_append_head_dims(dtype, kv_heads, cache_len, head_dim):
    """Explicit num_splits=1 keeps one block per cache head with the fused append."""
    torch.manual_seed(1)
    batch, heads = 2, 8
    q = torch.randn(batch, 1, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, 1, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    k_cache = torch.randn(
        batch, cache_len + 64, kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
    # Ragged lengths leave the last key tile partially filled.
    cache_seqlens = torch.tensor(
        [cache_len, cache_len - 37], device="cuda", dtype=torch.int32
    )
    options = {"k": k, "v": v, "seqlens_k": cache_seqlens, "num_splits": 1}
    actual = flash_attn2_rs.fwd_kvcache(q, k_cache, v_cache, **options)
    expected = flash_attn2.fwd_kvcache(q, k_cache_ref, v_cache_ref, **options)
    assert_all_close(actual, expected, atol=0.007, rtol=0.04)
    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
def test_varlen_causal_head_first_grid(dtype, paged, head_dim):
    """Causal packed prefill orders blocks head-first; odd lengths and tails."""
    torch.manual_seed(2)
    heads, kv_heads = 8, 2
    lengths = [777, 1025, 64, 300]
    total = sum(lengths)
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    options = dict(max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), is_causal=True)
    expected = flash_attn2.varlen_fwd(q, k, v, cu, cu, **options)
    if paged:
        pages = [(n + 255) // 256 for n in lengths]
        order = torch.randperm(sum(pages), dtype=torch.int32)
        table = torch.zeros(len(lengths), max(pages), dtype=torch.int32)
        kp = torch.zeros(sum(pages), 256, kv_heads, head_dim, device="cuda", dtype=dtype)
        vp = torch.zeros_like(kp)
        start, used = 0, 0
        for row, (n, count) in enumerate(zip(lengths, pages)):
            table[row, :count] = order[used : used + count]
            for i, page in enumerate(order[used : used + count].tolist()):
                rows = min(256, n - i * 256)
                kp[page, :rows] = k[start + i * 256 : start + i * 256 + rows]
                vp[page, :rows] = v[start + i * 256 : start + i * 256 + rows]
            start += n
            used += count
        actual = flash_attn2_rs.varlen_fwd(
            q, kp, vp, cu, cu, block_table=table.cuda(), **options
        )
    else:
        actual = flash_attn2_rs.varlen_fwd(q, k, v, cu, cu, **options)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_k", "scale"),
    [(1000, 1000, None), (1024, 777, 0.3), (65, 1024, None), (1024, 1030, None), (2049, 1100, 0.2)],
)
def test_dense_d32_raw_default_tile(dtype, seqlen_q, seqlen_k, scale):
    """Unmasked exact d32: 64x64 fused-scale tile and the resident-Q 128x64 tile."""
    torch.manual_seed(4)
    q = torch.randn(2, seqlen_q, 8, 32, device="cuda", dtype=dtype)
    k = torch.randn(2, seqlen_k, 2, 32, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected = flash_attn2.fwd(q, k, v, softmax_scale=scale)
    actual = flash_attn2_rs.fwd(q, k, v, softmax_scale=scale)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [(-1, 0), (300, 0), (100, 50)])
@pytest.mark.parametrize(("seqlen_q", "seqlen_k", "head_dim"), [(1000, 1000, 32), (333, 1030, 24), (2048, 2048, 32)])
def test_dense_d32_masked_wide_tile(dtype, window, seqlen_q, seqlen_k, head_dim):
    """Masked small heads use 128-key tiles; ragged tails and GQA stay exact."""
    torch.manual_seed(3)
    q = torch.randn(2, seqlen_q, 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, seqlen_k, 2, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    options = dict(window_size_left=window[0], window_size_right=window[1], is_causal=window == (-1, 0))
    expected = flash_attn2.fwd(q, k, v, **options)
    actual = flash_attn2_rs.fwd(q, k, v, **options)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
def test_kvcache_grouped_split_head_dims(head_dim):
    """Automatic splits group GQA heads as query rows for every head-dim bucket."""
    torch.manual_seed(0)
    q = torch.randn(2, 1, 8, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 1, 2, head_dim, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    k_cache = torch.randn(2, 4096, 2, head_dim, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    k_cache_ref, v_cache_ref = k_cache.clone(), v_cache.clone()
    cache_seqlens = torch.tensor([4000, 1000], device="cuda", dtype=torch.int32)
    options = {"k": k, "v": v, "cache_seqlens": cache_seqlens, "num_splits": 0}
    actual = flash_attn2_rs.flash_attn_with_kvcache(q, k_cache, v_cache, **options)
    expected = flash_attn2.flash_attn_with_kvcache(
        q, k_cache_ref, v_cache_ref, **options
    )
    torch.testing.assert_close(actual, expected, atol=0.007, rtol=0.04)
    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("seqlen", [1000, 4096])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
def test_paged_causal_prefill_head_dims(dtype, kv_heads, seqlen, head_dim):
    """Causal prefill over shuffled 256-row pages, including a ragged last page."""
    torch.manual_seed(0)
    heads = 8
    lengths = [seqlen, seqlen - 333]
    pages_per_sequence = [(n + 255) // 256 for n in lengths]
    order = torch.randperm(sum(pages_per_sequence), dtype=torch.int32)
    block_table = torch.zeros(len(lengths), max(pages_per_sequence), dtype=torch.int32)
    start = 0
    for row, count in enumerate(pages_per_sequence):
        block_table[row, :count] = order[start : start + count]
        start += count
    block_table = block_table.cuda()
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    q = torch.randn(cu_seqlens[-1], heads, head_dim, device="cuda", dtype=dtype)
    k_pages = torch.randn(len(order), 256, kv_heads, head_dim, device="cuda", dtype=dtype)
    v_pages = torch.randn_like(k_pages)
    args = (q, k_pages, v_pages, cu_seqlens, cu_seqlens)
    options = {
        "block_table": block_table,
        "max_seqlen_q": seqlen,
        "max_seqlen_k": seqlen,
        "is_causal": True,
    }
    actual = flash_attn2_rs.varlen_fwd(*args, **options)
    expected = flash_attn2.varlen_fwd(*args, **options)
    assert_all_close(actual[:2], expected[:2], atol=0.005, rtol=0.03)


def case(name, batch, num_heads, num_kv_heads, seqlen_q, seqlen_k, **options):
    shape = (batch, num_heads, num_kv_heads, seqlen_q, seqlen_k)
    return pytest.param(shape, options, id=name)


PARITY_CASES = [
    case("full-mha", 2, 4, 4, 256, 256),
    case("causal-mha", 2, 4, 4, 256, 256, causal=True),
    case("causal-gqa-h8-hk2", 2, 8, 2, 256, 256, causal=True),
    case("full-mqa-h4-hk1", 1, 4, 1, 192, 192),
    case("causal-sq256-sk512", 1, 4, 4, 256, 512, causal=True),
    case("custom-scale", 2, 4, 4, 256, 256, causal=True, softmax_scale=0.05),
    case("local-causal-wl64", 2, 4, 4, 256, 256, causal=True, window_left=64),
    case("window-wl64-wr64", 1, 4, 4, 256, 256, window_left=64, window_right=64),
    case("softcap30-causal", 2, 4, 4, 256, 256, causal=True, softcap=30.0),
    case("softcap50-full", 1, 4, 4, 192, 192, softcap=50.0),
    case("alibi-causal", 2, 4, 4, 256, 256, causal=True, alibi=True),
    case("alibi-full", 1, 4, 4, 192, 192, alibi=True),
    case(
        "alibi-local-wl64", 1, 4, 4, 256, 256, causal=True, window_left=64, alibi=True
    ),
    case(
        "softcap-alibi-gqa-causal",
        2,
        8,
        2,
        256,
        256,
        causal=True,
        softcap=20.0,
        alibi=True,
    ),
    case("long-causal-s2048", 1, 8, 8, 2048, 2048, causal=True),
    # Sliding windows: asymmetric, sub-tile, sq != sk and leading-tile skip.
    case("window-wr32", 1, 4, 4, 256, 256, window_right=32),
    case("window-wl32", 1, 4, 4, 256, 256, window_left=32),
    case("tiny-window-causal-wl16", 1, 4, 4, 256, 256, causal=True, window_left=16),
    case("zero-window-causal-wl0", 1, 4, 4, 256, 256, causal=True, window_left=0),
    case("tile-edge-window-wl64-s512", 1, 4, 4, 512, 512, causal=True, window_left=64),
    case(
        "tile-edge-window-wl128-s512", 1, 4, 4, 512, 512, causal=True, window_left=128
    ),
    case("window-wl200-wr8", 1, 4, 4, 512, 512, window_left=200, window_right=8),
    case("window-sq128-sk384-wl64", 1, 4, 4, 128, 384, causal=True, window_left=64),
    case("window-sq384-sk256-wl64", 1, 4, 4, 384, 256, causal=True, window_left=64),
    case("odd-window-s333-wl70", 1, 4, 4, 333, 333, causal=True, window_left=70),
    case("long-window-s2048-wl256", 1, 8, 8, 2048, 2048, causal=True, window_left=256),
    case(
        "long-window-bf16-wl512",
        1,
        4,
        4,
        2048,
        2048,
        causal=True,
        window_left=512,
        dtype=torch.bfloat16,
    ),
    case(
        "d64-window-causal-wl96",
        1,
        4,
        4,
        512,
        512,
        causal=True,
        window_left=96,
        head_dim=64,
    ),
    case(
        "window-alibi-gqa-wl100",
        2,
        8,
        2,
        512,
        512,
        causal=True,
        window_left=100,
        alibi=True,
    ),
    case("odd-full-s200", 2, 4, 4, 200, 200),
    case("odd-causal-s333", 2, 4, 4, 333, 333, causal=True),
    case("odd-causal-s1000", 1, 8, 8, 1000, 1000, causal=True),
    case("odd-gqa-causal-s130", 2, 8, 2, 130, 130, causal=True),
    case("odd-sq100-sk250", 1, 4, 4, 100, 250, causal=True),
    case(
        "odd-window-softcap-s300",
        1,
        4,
        4,
        300,
        300,
        causal=True,
        window_left=100,
        softcap=30.0,
    ),
    case("odd-alibi-full-s150", 1, 4, 4, 150, 150, alibi=True),
    # Head-dim buckets (32/64/96/192/256; 128 is covered above).
    *[
        param
        for head_dim in (32, 64, 96, 192, 256)
        for param in (
            case(f"d{head_dim}-full-mha", 2, 4, 4, 256, 256, head_dim=head_dim),
            case(
                f"d{head_dim}-causal-mha",
                2,
                4,
                4,
                256,
                256,
                causal=True,
                head_dim=head_dim,
            ),
            case(
                f"d{head_dim}-causal-gqa",
                2,
                8,
                2,
                256,
                256,
                causal=True,
                head_dim=head_dim,
            ),
            case(
                f"d{head_dim}-softcap-causal",
                1,
                4,
                4,
                192,
                192,
                causal=True,
                softcap=30.0,
                head_dim=head_dim,
            ),
            case(
                f"d{head_dim}-alibi-causal",
                2,
                4,
                4,
                256,
                256,
                causal=True,
                alibi=True,
                head_dim=head_dim,
            ),
            case(
                f"d{head_dim}-odd-causal-s333",
                1,
                4,
                4,
                333,
                333,
                causal=True,
                head_dim=head_dim,
            ),
        )
    ],
    *[
        case(
            f"d{head_dim}-bf16-causal",
            batch,
            4,
            4,
            256,
            256,
            causal=True,
            dtype=torch.bfloat16,
            head_dim=head_dim,
        )
        for batch, head_dim in ((2, 32), (2, 64), (1, 256))
    ],
    # Non-bucket head dims run zero-filled in the next compiled bucket.
    *[
        case(
            f"d{head_dim}-causal-mha", 2, 4, 4, 256, 256, causal=True, head_dim=head_dim
        )
        for head_dim in (8, 48, 80, 104, 160, 224)
    ],
    case("d72-gqa-causal", 2, 8, 2, 200, 200, causal=True, head_dim=72),
    case("d144-alibi-causal", 1, 4, 4, 192, 192, causal=True, alibi=True, head_dim=144),
    case(
        "d200-bf16-causal",
        1,
        4,
        4,
        256,
        256,
        causal=True,
        dtype=torch.bfloat16,
        head_dim=200,
    ),
    case("bf16-full-mha", 2, 4, 4, 256, 256, dtype=torch.bfloat16),
    case("bf16-causal-mha", 2, 4, 4, 256, 256, causal=True, dtype=torch.bfloat16),
    case("bf16-causal-gqa", 2, 8, 2, 256, 256, causal=True, dtype=torch.bfloat16),
    case(
        "bf16-softcap-causal",
        2,
        4,
        4,
        256,
        256,
        causal=True,
        softcap=30.0,
        dtype=torch.bfloat16,
    ),
    case(
        "bf16-alibi-causal",
        2,
        4,
        4,
        256,
        256,
        causal=True,
        alibi=True,
        dtype=torch.bfloat16,
    ),
    case(
        "bf16-local-wl64",
        2,
        4,
        4,
        256,
        256,
        causal=True,
        window_left=64,
        dtype=torch.bfloat16,
    ),
    case("bf16-odd-causal-s333", 2, 4, 4, 333, 333, causal=True, dtype=torch.bfloat16),
    case(
        "bf16-long-causal-s2048", 1, 8, 8, 2048, 2048, causal=True, dtype=torch.bfloat16
    ),
]


PARITY_CASES += [
    case(
        f"d192-dense-{dtype}-{length}",
        2, 8, 8, 1024, length,
        head_dim=192,
        dtype=dtype,
        softmax_scale=0.125 if length == 512 else None,
    )
    for dtype in (torch.float16, torch.bfloat16)
    for length in (64, 512, 1024)
]


PARITY_CASES += [
    case(
        f"d32-cross-splits-{dtype}-{length}-{causal}",
        1, 4, 2, 64, length,
        causal=causal,
        dtype=dtype,
        head_dim=32,
    )
    for dtype in (torch.float16, torch.bfloat16)
    for length in (1024, 2048)
    for causal in (False, True)
]


PARITY_CASES += [
    case(
        f"d32-cross-three-splits-{dtype}",
        16, 8, 2, 64, 4096,
        dtype=dtype,
        head_dim=32,
    )
    for dtype in (torch.float16, torch.bfloat16)
]


@pytest.mark.kernels_ci
@pytest.mark.parametrize(("shape", "options"), PARITY_CASES)
def test_forward_parity(shape, options):
    """Forward output and LSE match the reference across masks and head sizes."""
    batch, num_heads, num_kv_heads, seqlen_q, seqlen_k = shape
    options = dict(options)
    dtype = options.pop("dtype", torch.float16)
    head_dim = options.pop("head_dim", 128)
    alibi = options.pop("alibi", False)

    torch.manual_seed(0)
    q = torch.randn(batch, seqlen_q, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    if alibi:
        heads = torch.arange(num_heads, device="cuda").float()
        options["alibi_slopes"] = (2.0 ** (-8.0 * (heads + 1) / num_heads)).contiguous()

    out, lse = fwd_explicit(q, k, v, **options)
    expected_out, expected_lse = reference_forward(q, k, v, **options)

    # bf16 has fewer mantissa bits than f16, so bounds are looser.
    if dtype == torch.float16:
        atol, rtol, mean_atol, lse_atol = 2e-3, 2.5e-2, 3e-3, 3e-2
    else:
        atol, rtol, mean_atol, lse_atol = 1e-2, 5e-2, 8e-3, 8e-2
    out, expected_out = out.float(), expected_out.float()
    assert torch.isfinite(out).all()
    finite = torch.isfinite(expected_out)
    torch.testing.assert_close(out[finite], expected_out[finite], atol=atol, rtol=rtol)
    assert (out[finite] - expected_out[finite]).abs().mean() < mean_atol
    # Compare nonfinite entries too so missing LSE values are not hidden.
    torch.testing.assert_close(lse.float(), expected_lse.float(), atol=lse_atol, rtol=0)


def test_strided_views():
    """Zero-copy [B, H, S, D] transposes match contiguous reference inputs."""
    torch.manual_seed(0)
    q = torch.randn(2, 8, 256, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, 256, 128, device="cuda", dtype=q.dtype)
    v = torch.randn(2, 2, 256, 128, device="cuda", dtype=q.dtype)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    out, lse = fwd_explicit(q, k, v, causal=True)
    expected_out, expected_lse = reference_forward(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=True
    )
    torch.testing.assert_close(out.float(), expected_out.float(), atol=2e-2, rtol=0)
    torch.testing.assert_close(lse.float(), expected_lse.float(), atol=3e-2, rtol=0)


# The head size must be at most 256 and a multiple of 8, like the reference.
@pytest.mark.parametrize("head_size", [264, 100], ids=["over-256", "not-multiple-of-8"])
def test_rejects_head_size(head_size):
    q = torch.randn(1, 128, 4, head_size, device="cuda", dtype=torch.float16)
    with pytest.raises((ValueError, TypeError)):
        fwd_explicit(q, q, q)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", [1025, 2048, 2049, 4095, 4096])
def test_backward_long_d64(dtype, causal, seqlen):
    """Long fused MHA handles tile tails and matches the reference gradients."""
    torch.manual_seed(948)
    q = torch.randn(2, seqlen, 8, 64, device="cuda", dtype=dtype)
    k, v, dout = (torch.randn_like(q) for _ in range(3))
    saved = flash_attn2.fwd(q, k, v, is_causal=causal)
    args = (dout, q, k, v, saved[0], saved[1])
    expected = flash_attn2.bwd(*args, is_causal=causal, deterministic=True)
    actual = flash_attn2_rs.bwd(*args, is_causal=causal)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)
    # Deterministic requests must retain the non-atomic fallback.
    deterministic = flash_attn2_rs.bwd(*args, is_causal=causal, deterministic=True)
    repeated = flash_attn2_rs.bwd(*args, is_causal=causal, deterministic=True)
    assert_all_close(deterministic[:3], expected[:3], atol=atol, rtol=rtol)
    for first, second in zip(deterministic[:3], repeated[:3]):
        assert torch.equal(first, second)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("zero_tensors", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([0, 129, 1025, 257], [17, 0, 1023, 513]),
        ([0, 1024, 1023, 129], [17, 2048, 1023, 0]),
        ([2048, 2047], [4096, 4095]),
        ([129, 256, 0, 512], [256, 257, 0, 512]),
        ([1, 31, 32, 33, 128], [7, 0, 64, 129, 128]),
        ([129, 1024, 0, 1023], [129, 1024, 0, 1024]),
        ([129, 2048, 2047], [129, 4096, 4095]),
    ],
    ids=["compact", "query-pipeline", "wide-query-pipeline", "unaligned-full-tiles", "padded-short-tails", "unaligned-query-tiles", "unaligned-wide-tiles"],
)
def test_varlen_backward_combined_d64(
    dtype, causal, zero_tensors, deterministic, qlens, klens
):
    """Combined dKV respects packed offsets, tails, empty queries and keys."""
    torch.manual_seed(948)
    cuq = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    cuk = torch.tensor(
        [0, *torch.tensor(klens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(qlens), 8, 64, device="cuda", dtype=dtype)
    k = torch.randn(sum(klens), 8, 64, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    options = dict(
        max_seqlen_q=max(qlens),
        max_seqlen_k=max(klens),
        is_causal=causal,
        zero_tensors=zero_tensors,
    )
    saved = flash_attn2.varlen_fwd(q, k, v, cuq, cuk, **options)
    args = (dout, q, k, v, saved[0], saved[1], cuq, cuk)
    expected = flash_attn2.varlen_bwd(*args, deterministic=True, **options)
    actual = flash_attn2_rs.varlen_bwd(*args, deterministic=deterministic, **options)
    repeated = flash_attn2_rs.varlen_bwd(*args, deterministic=deterministic, **options)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)
    assert_all_close(repeated[:3], expected[:3], atol=atol, rtol=rtol)
    if deterministic:
        for first, second in zip(actual[:3], repeated[:3]):
            assert torch.equal(first, second)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [1025, 4097])
@pytest.mark.parametrize("scale", [0.125, 0.3])
def test_forward_d64_unmasked_tiles(dtype, seqlen, scale):
    """Raw-score tiles preserve strided inputs, tail masking, and LSE scaling."""
    torch.manual_seed(948)
    q = torch.randn(2, seqlen * 2, 8, 64, device="cuda", dtype=dtype)[:, ::2]
    k = torch.randn(2, seqlen + 6, 8, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected = flash_attn2.fwd(q, k, v, softmax_scale=scale)
    actual = flash_attn2_rs.fwd(q, k, v, softmax_scale=scale)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("seqlen_q", "seqlen_k", "kv_heads", "strided"),
    [
        (1024, 1024, 8, False),
        (1024, 4096, 8, False),
        (2048, 1152, 8, False),
        (1152, 320, 8, False),
        (384, 4096, 8, False),
        (1024, 1000, 8, False),
        (1000, 1024, 8, False),
        (1025, 1088, 8, False),
        (1024, 1024, 2, False),
        (1024, 1024, 8, True),
    ],
)
@pytest.mark.parametrize("scale", [None, 0.3])
@pytest.mark.parametrize("head_dim", [32, 96, 256])
def test_forward_resident_dense_tile(
    dtype, seqlen_q, seqlen_k, kv_heads, strided, scale, head_dim
):
    """Exact unmasked d32/d96/d256 use the resident-Q tiles only for full
    tiles, contiguous MHA and full waves; ragged lengths, small grids, GQA and
    strided inputs fall back."""
    torch.manual_seed(head_dim)
    q = torch.randn(2, seqlen_q, 8, head_dim, device="cuda", dtype=dtype)
    if strided:
        q = torch.randn(2, seqlen_q, 8, 2 * head_dim, device="cuda", dtype=dtype)
        q = q[..., :head_dim]
    k = torch.randn(2, seqlen_k, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected = flash_attn2.fwd(q, k, v, softmax_scale=scale)
    actual = flash_attn2_rs.fwd(q, k, v, softmax_scale=scale)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("page_size", [256, 512, 768])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_forward_page_sizes(page_size, dtype):
    """Power-of-two page addressing and its fallback match packed attention."""
    torch.manual_seed(948)
    q = torch.randn(386, 8, 64, device="cuda", dtype=dtype)
    kp = torch.randn(4, page_size, 2, 64, device="cuda", dtype=dtype)
    vp = torch.randn_like(kp)
    table = torch.tensor([[2, 0], [1, 3]], device="cuda", dtype=torch.int32)
    cuq = torch.tensor([0, 257, 386], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, page_size + 33, 3 * page_size + 32], device="cuda", dtype=torch.int32)
    k, v = (torch.cat((p[2], p[0, :33], p[1], p[3, :-1])) for p in (kp, vp))
    options = dict(max_seqlen_q=257, max_seqlen_k=2 * page_size - 1, is_causal=True)
    expected = flash_attn2.varlen_fwd(q, k, v, cuq, cuk, **options)
    actual = flash_attn2_rs.varlen_fwd(q, kp, vp, cuq, cuk, block_table=table, **options)
    atol, rtol = (0.04, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_length", "heads", "scale"),
    [(1, 4, None), (1, 8, 0.3), (1, 16, None), (1, 8, 0.0),
     (1, 8, -0.1), (7, 8, None), (16, 8, None), (17, 8, None)],
)
@pytest.mark.parametrize("append", [False, True])
def test_cached_d128_short_queries(dtype, query_length, heads, scale, append):
    """Underfilled cache tiles handle partial warps, key tails and in-place append."""
    torch.manual_seed(442)
    q = torch.randn(2, heads, query_length, 128, device="cuda", dtype=dtype).transpose(1, 2)
    kcache = torch.randn(2, 1025, 2, 128, device="cuda", dtype=dtype)
    vcache = torch.randn_like(kcache)
    reference_k, reference_v = kcache.clone(), vcache.clone()
    lengths = torch.tensor(
        [1025 - query_length, 1007 - query_length], device="cuda", dtype=torch.int32
    )
    k = torch.randn(2, query_length, 2, 128, device="cuda", dtype=dtype) if append else None
    v = torch.randn_like(k) if append else None
    options = dict(k=k, v=v, seqlens_k=lengths, num_splits=1, softmax_scale=scale)
    if scale is not None and scale <= 0.0:
        # v4 produces NaNs for nonpositive scales with padded cache keys.
        # Verify the generic fallback against direct attention instead.
        expected_out, expected_lse = [], []
        for batch in range(2):
            start = int(lengths[batch])
            if append:
                reference_k[batch, start:start + query_length].copy_(k[batch])
                reference_v[batch, start:start + query_length].copy_(v[batch])
            valid = start + (query_length if append else 0)
            keys = reference_k[batch, :valid].float().repeat_interleave(heads // 2, dim=1)
            values = reference_v[batch, :valid].float().repeat_interleave(heads // 2, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q[batch].float(), keys) * scale
            expected_out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(dtype))
            expected_lse.append(scores.logsumexp(-1))
        expected = [torch.stack(expected_out), torch.stack(expected_lse)]
    else:
        expected = flash_attn2.fwd_kvcache(q, reference_k, reference_v, **options)
    actual = flash_attn2_rs.fwd_kvcache(q, kcache, vcache, **options)
    atol, rtol = (0.03, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)
    assert torch.equal(kcache, reference_k)
    assert torch.equal(vcache, reference_v)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grouped", [False, True])
def test_d128_fused_append_snapshots_overlapping_sources(dtype, grouped):
    """Append must preserve Q and source K/V even when destination rows overlap them."""
    torch.manual_seed(448)
    kcache = torch.randn(1, 1025, 2, 128, device="cuda", dtype=dtype)
    vcache = torch.randn_like(kcache)
    q = kcache[:, :4].reshape(1, 1, 8, 128) if grouped else kcache[:, 3:4]
    k, v = kcache[:, 4:6], vcache[:, 4:6]
    lengths = torch.tensor([3], device="cuda", dtype=torch.int32)
    kr, vr = kcache.clone(), vcache.clone()
    expected = flash_attn2.fwd_kvcache(
        q.clone(), kr, vr, k=k.clone(), v=v.clone(), seqlens_k=lengths, num_splits=1
    )
    actual = flash_attn2_rs.fwd_kvcache(
        q, kcache, vcache, k=k, v=v, seqlens_k=lengths, num_splits=1
    )
    atol, rtol = (0.03, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)
    assert torch.equal(kcache, kr)
    assert torch.equal(vcache, vr)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("new_keys", [1, 2, 7, 17])
def test_d128_fused_append_strided_sources(dtype, new_keys):
    """A single query can append several strided K/V rows up to the cache tail."""
    torch.manual_seed(449)
    q = torch.randn(2, 8, 1, 128, device="cuda", dtype=dtype).transpose(1, 2)
    kc = torch.randn(2, 1025, 2, 128, device="cuda", dtype=dtype)
    vc = torch.randn_like(kc)
    k = torch.randn(2, 2, new_keys, 128, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(2, 2, new_keys, 128, device="cuda", dtype=dtype).transpose(1, 2)
    lengths = torch.tensor([1025 - new_keys, 1008 - new_keys], device="cuda", dtype=torch.int32)
    kr, vr = kc.clone(), vc.clone()
    options = dict(k=k, v=v, seqlens_k=lengths, num_splits=1)
    expected = flash_attn2.fwd_kvcache(q, kr, vr, **options)
    actual = flash_attn2_rs.fwd_kvcache(q, kc, vc, **options)
    atol, rtol = (0.03, 0.04) if dtype == torch.bfloat16 else (0.005, 0.03)
    assert_all_close(actual[:2], expected[:2], atol=atol, rtol=rtol)
    assert torch.equal(kc, kr)
    assert torch.equal(vc, vr)


@pytest.mark.kernels_ci
def test_d128_fused_append_rejects_aliased_metadata_without_mutation():
    q = torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.float16)
    kc = torch.randn(1, 1025, 1, 128, device="cuda", dtype=q.dtype)
    vc = torch.randn_like(kc)
    lengths = kc.view(torch.int32).flatten()[:1]
    lengths.fill_(3)
    saved_k, saved_v = kc.clone(), vc.clone()
    versions = kc._version, vc._version
    with pytest.raises((ValueError, RuntimeError), match="disjoint"):
        flash_attn2_rs.fwd_kvcache(q, kc, vc, k=q, v=q, seqlens_k=lengths, num_splits=1)
    assert (kc._version, vc._version) == versions
    assert torch.equal(kc, saved_k)
    assert torch.equal(vc, saved_v)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offsets", [(0, 0), (0, 2), (2, 0), (2, 2)])
@pytest.mark.parametrize("dropout", [0.0, 0.17])
def test_preprocess_alignment(dtype, offsets, dropout):
    torch.manual_seed(981)
    tensors = [
        torch.randn(2, 129, 3, 64 + offset, device="cuda", dtype=dtype)[..., offset:]
        for offset in offsets
    ]
    dout, out = tensors
    actual = flash_attn2_rs.ops.bwd_preprocess(dout, out, dropout)
    expected = (dout.float() * out.float()).sum(-1).transpose(1, 2) * (1.0 - dropout)
    torch.testing.assert_close(actual[..., :129], expected, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(actual[..., 129:]).item() == 0
    repeat = flash_attn2_rs.ops.bwd_preprocess(dout, out, dropout)
    torch.testing.assert_close(actual, repeat, atol=0, rtol=0)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [2, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_backward_d128_warp_layout(dtype, causal, kv_heads, strided):
    torch.manual_seed(134)
    q = torch.randn(2, 512, 8, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 512, kv_heads, 128, device="cuda", dtype=dtype)
    v, dout = torch.randn_like(k), torch.randn_like(q)
    if strided:
        q, k, v, dout = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v, dout)]
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, is_causal=causal)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, is_causal=causal)
    expected = flash_attn2.bwd(*args, rng_state=rng, is_causal=causal, deterministic=True)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)


@pytest.mark.kernels_ci
@requires_philox_state
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [256, 1024, 2048])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_d64_permuted_accumulator(dtype, length, strided, causal):
    torch.manual_seed(135)
    q, k, v, dout = [torch.randn(2, length, 8, 64, device="cuda", dtype=dtype) for _ in range(4)]
    if strided:
        q, k, v, dout = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v, dout)]
    out, lse, _, rng = flash_attn2_rs.fwd(q, k, v, is_causal=causal)
    args = (dout, q, k, v, out, lse)
    actual = flash_attn2_rs.bwd(*args, rng_state=rng, is_causal=causal)
    expected = flash_attn2.bwd(*args, rng_state=rng, deterministic=True, is_causal=causal)
    atol, rtol = (0.06, 0.08) if dtype == torch.bfloat16 else (0.008, 0.04)
    assert_all_close(actual[:3], expected[:3], atol=atol, rtol=rtol)
