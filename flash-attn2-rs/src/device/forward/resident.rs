// Dense MHA with Q resident in shared memory; host guards full tiles, plain
// attention and packed exact-width layouts. K and V are single buffered: K
// restages after QK^T, V after PV, each behind the other's wait. Instantiated
// for the reference shapes of D96 (128x64, four warps), D32 (128x128, eight
// warps) and D256 (64x64, four warps).
use crate::device::dropout::Dropout;
use crate::device::forward::LOG2_E;
use crate::device::forward::dense::FlashSpec;
use crate::device::inlined::{
    cvt, fast_exp2, fmax, ldmatrix_x4_shared, ldmatrix_x4_trans_shared, lg2_approx, mma,
    shared_address,
};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{ptx_asm, thread};

// Evaluate the target name at compile time; device code cannot carry strings.
const HAS_L2_PREFETCH: bool = match option_env!("CUDA_OXIDE_TARGET") {
    Some(target) => {
        let bytes = target.as_bytes();
        let mut arch = 0u32;
        let mut i = 3;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            arch = arch * 10 + (bytes[i] - b'0') as u32;
            i += 1;
        }
        arch >= 89
    }
    None => false,
};

/// Planes of `PW` u32 columns: 64-byte planes (PW 16) use the reference
/// Swizzle<2,3,3>, where the 16-byte chunk XORs row bits 1-2; 128-byte planes
/// (PW 32) use Swizzle<3,3,3> with row bits 0-2. Either way eight consecutive
/// rows hit distinct banks for ldmatrix and for the rows written by cp.async.
#[inline(always)]
fn index<const ROWS: u32, const PW: u32>(row: u32, col: u32) -> u32 {
    (col / PW) * (ROWS * PW) + row * PW + ((col % PW) ^ (swizzle::<PW>(row) * 4))
}

#[inline(always)]
fn swizzle<const PW: u32>(row: u32) -> u32 {
    if PW == 32 { row & 7 } else { (row >> 1) & 3 }
}

#[inline(always)]
unsafe fn cp_async_16(dst: u32, src: *const u32) {
    unsafe {
        // Full tiles are guaranteed by dispatch. Stream through L2 with a
        // 128-byte fetch hint; these copies do not need L1 allocation.
        // The backend emits PTX 7.0 for sm_80, before L2 prefetch hints.
        if HAS_L2_PREFETCH {
            ptx_asm!(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;",
                in("r") dst,
                in("l") src as u64,
                clobber("memory")
            );
        } else {
            ptx_asm!(
                "cp.async.cg.shared.global [%0], [%1], 16;",
                in("r") dst,
                in("l") src as u64,
                clobber("memory")
            );
        }
    }
}

/// Every step copies THREADS 16-byte chunks: PW / 4 consecutive threads fill
/// one plane row, so a step covers 4 THREADS / PW rows of one plane. The
/// per-thread row and swizzle are fixed; steps add constants only.
#[inline(always)]
unsafe fn stage<const ROWS: u32, const PW: u32, const THREADS: u32, const STEPS: u32>(
    dst: u32,
    src: *const u32,
    stride: u32,
    tid: u32,
) {
    unsafe {
        let r = tid / (PW / 4);
        let chunk = (tid % (PW / 4)) * 4;
        // Rows advance by multiples of 16 per step, which leaves bits 0-2 alone.
        let dst = dst + (r * PW + (chunk ^ (swizzle::<PW>(r) * 4))) * 4;
        let src = src.add((r * stride + chunk) as usize);
        let mut step = 0;
        while step < STEPS {
            thread::__unroll_config::<0>();
            let rows = step * (4 * THREADS / PW);
            let plane = rows / ROWS;
            let row = rows % ROWS;
            cp_async_16(
                dst + (plane * ROWS * PW + row * PW) * 4,
                src.add((row * stride + plane * PW) as usize),
            );
            step += 1;
        }
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

/// Online softmax on raw scores: the exponent folds the log2 scale into one
/// FMA. Unmasked dense rows always hold finite scores, so no -inf guard.
/// `MODE` bit 0 reduces the row maximum as a tree, bit 1 rescales the output
/// without a branch.
#[inline(always)]
fn softmax<
    const BF16: bool,
    const RF: usize,
    const NKT: usize,
    const ND: usize,
    const SF: usize,
    const OF: usize,
    const MODE: u32,
>(
    s: &[[f32; 4]; SF],
    p: &mut [[u32; 2]; SF],
    out: &mut [[f32; 4]; OF],
    m_row: &mut [f32; 4],
    l_row: &mut [f32; 4],
    scale: f32,
) {
    let mut rf = 0usize;
    while rf < RF {
        thread::__unroll_config::<0>();
        let mut cm0;
        let mut cm1;
        if MODE & 1 != 0 {
            // Pairwise tree: a log-depth dependency chain instead of 2 NKT.
            let mut m = [[0.0f32; 2]; NKT];
            let mut nt = 0usize;
            while nt < NKT {
                thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                m[nt] = [fmax(f[0], f[1]), fmax(f[2], f[3])];
                nt += 1;
            }
            // Literal bounds per stage: the unroller rejects a halving step.
            if NKT == 16 {
                let mut nt = 0usize;
                while nt < 8 {
                    thread::__unroll_config::<0>();
                    m[nt] = [fmax(m[nt][0], m[nt + 8][0]), fmax(m[nt][1], m[nt + 8][1])];
                    nt += 1;
                }
            }
            let mut nt = 0usize;
            while nt < 4 {
                thread::__unroll_config::<0>();
                m[nt] = [fmax(m[nt][0], m[nt + 4][0]), fmax(m[nt][1], m[nt + 4][1])];
                nt += 1;
            }
            m[0] = [fmax(m[0][0], m[2][0]), fmax(m[0][1], m[2][1])];
            m[1] = [fmax(m[1][0], m[3][0]), fmax(m[1][1], m[3][1])];
            cm0 = fmax(m[0][0], m[1][0]);
            cm1 = fmax(m[0][1], m[1][1]);
        } else {
            cm0 = f32::NEG_INFINITY;
            cm1 = f32::NEG_INFINITY;
            let mut nt = 0usize;
            while nt < NKT {
                thread::__unroll_config::<0>();
                let f = &s[rf * NKT + nt];
                cm0 = fmax(fmax(cm0, f[0]), f[1]);
                cm1 = fmax(fmax(cm1, f[2]), f[3]);
                nt += 1;
            }
        }
        cm0 = fmax(cm0, shuffle_xor_f32(cm0, 1));
        cm0 = fmax(cm0, shuffle_xor_f32(cm0, 2));
        cm1 = fmax(cm1, shuffle_xor_f32(cm1, 1));
        cm1 = fmax(cm1, shuffle_xor_f32(cm1, 2));
        let m_new0 = fmax(m_row[rf * 2], cm0);
        let m_new1 = fmax(m_row[rf * 2 + 1], cm1);
        let alpha0 = exp2::<MODE>((m_row[rf * 2] - m_new0) * scale);
        let alpha1 = exp2::<MODE>((m_row[rf * 2 + 1] - m_new1) * scale);
        m_row[rf * 2] = m_new0;
        m_row[rf * 2 + 1] = m_new1;
        if MODE & 2 != 0 || alpha0 != 1.0 || alpha1 != 1.0 {
            let mut d = 0usize;
            while d < ND {
                thread::__unroll_config::<0>();
                let f = &mut out[rf * ND + d];
                f[0] *= alpha0;
                f[1] *= alpha0;
                f[2] *= alpha1;
                f[3] *= alpha1;
                d += 1;
            }
        }
        let bias0 = -m_new0 * scale;
        let bias1 = -m_new1 * scale;
        let mut cs0 = 0.0f32;
        let mut cs1 = 0.0f32;
        let mut nt = 0usize;
        while nt < NKT {
            thread::__unroll_config::<0>();
            let f = &s[rf * NKT + nt];
            let e0 = exp2::<MODE>(fma(f[0], scale, bias0));
            let e1 = exp2::<MODE>(fma(f[1], scale, bias0));
            let e2 = exp2::<MODE>(fma(f[2], scale, bias1));
            let e3 = exp2::<MODE>(fma(f[3], scale, bias1));
            cs0 += e0 + e1;
            cs1 += e2 + e3;
            p[rf * NKT + nt] = [cvt::<BF16>(e0, e1), cvt::<BF16>(e2, e3)];
            nt += 1;
        }
        // Partial denominators; reduced across the quad in the epilogue.
        l_row[rf * 2] = l_row[rf * 2] * alpha0 + cs0;
        l_row[rf * 2 + 1] = l_row[rf * 2 + 1] * alpha1 + cs1;
        rf += 1;
    }
}

/// `MODE` bit 2 flushes denormals like the reference's fast-math exp2f, which
/// drops the predicate and fix-up multiply ptxas wraps around every MUFU.
#[inline(always)]
fn exp2<const MODE: u32>(x: f32) -> f32 {
    if MODE & 4 != 0 {
        let y: f32;
        unsafe {
            ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") y, in("f") x, options(register_only));
        }
        y
    } else {
        fast_exp2(x)
    }
}

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") value,
            in("f") a, in("f") b, in("f") c, options(register_only));
    }
    value
}

/// `ROWS` queries and `KEYS` per tile, `DU` u32 columns per row (head dim /
/// 2) in planes of `PW` columns, `RF` row fragments (16 rows each) per warp
/// with `THREADS` = ROWS / (16 RF) warps. Derived counts, spelled out because
/// loop bounds must be plain constants: `NKT` = KEYS / 8 score and `ND` = DU
/// / 4 output fragments per row fragment, `SF` = RF NKT, `OF` = RF ND, `KC` =
/// DU / 8 QK^T k-steps, `PVK` = KEYS / 16 PV k-steps, and `QSTEPS` = ROWS DU
/// / 4 THREADS, `KSTEPS` = KEYS DU / 4 THREADS copy steps. `MODE` selects
/// softmax variants (see `softmax`).
pub struct Resident<
    const BF16: bool,
    const ROWS: u32,
    const DU: u32,
    const PW: u32,
    const KEYS: u32,
    const RF: usize,
    const THREADS: u32,
    const NKT: usize,
    const ND: usize,
    const SF: usize,
    const OF: usize,
    const KC: u32,
    const PVK: u32,
    const QSTEPS: u32,
    const KSTEPS: u32,
    const MODE: u32,
>;

impl<
    const BF16: bool,
    const ROWS: u32,
    const DU: u32,
    const PW: u32,
    const KEYS: u32,
    const RF: usize,
    const THREADS: u32,
    const NKT: usize,
    const ND: usize,
    const SF: usize,
    const OF: usize,
    const KC: u32,
    const PVK: u32,
    const QSTEPS: u32,
    const KSTEPS: u32,
    const MODE: u32,
> FlashSpec
    for Resident<
        BF16,
        ROWS,
        DU,
        PW,
        KEYS,
        RF,
        THREADS,
        NKT,
        ND,
        SF,
        OF,
        KC,
        PVK,
        QSTEPS,
        KSTEPS,
        MODE,
    >
{
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        o: *mut u32,
        lse: *mut f32,
        _alibi: *const f32,
        scale: f32,
        _tanh_scale: f32,
        _window_left: i32,
        _window_right: i32,
        seqlen_q: u32,
        seqlen_k: u32,
        _has_alibi: u32,
        _alibi_causal: u32,
        h: u32,
        _gqa: u32,
        _hs_u32: u32,
        _q_bs: u32,
        _q_ss: u32,
        _q_hs: u32,
        _k_bs: u32,
        _k_ss: u32,
        _k_hs: u32,
        _v_bs: u32,
        _v_ss: u32,
        _v_hs: u32,
        _alibi_bs: u32,
        _dropout: Dropout,
    ) {
        const {
            assert!(RF as u32 * 16 * (THREADS / 32) == ROWS);
            assert!(NKT as u32 == KEYS / 8 && ND as u32 == DU / 4);
            assert!(SF == RF * NKT && OF == RF * ND);
            assert!(KC == DU / 8 && PVK == KEYS / 16);
            assert!((PW == 16 || PW == 32) && DU % PW == 0 && (4 * THREADS / PW) % 8 == 0);
            assert!(QSTEPS * THREADS * 4 == ROWS * DU && KSTEPS * THREADS * 4 == KEYS * DU);
        }
        unsafe {
            run::<
                BF16,
                ROWS,
                DU,
                PW,
                KEYS,
                RF,
                THREADS,
                NKT,
                ND,
                SF,
                OF,
                KC,
                PVK,
                QSTEPS,
                KSTEPS,
                MODE,
            >(q, k, v, o, lse, scale, seqlen_q, seqlen_k, h);
        }
    }
}

#[inline(always)]
unsafe fn run<
    const BF16: bool,
    const ROWS: u32,
    const DU: u32,
    const PW: u32,
    const KEYS: u32,
    const RF: usize,
    const THREADS: u32,
    const NKT: usize,
    const ND: usize,
    const SF: usize,
    const OF: usize,
    const KC: u32,
    const PVK: u32,
    const QSTEPS: u32,
    const KSTEPS: u32,
    const MODE: u32,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    o: *mut u32,
    lse: *mut f32,
    scale: f32,
    sq: u32,
    sk: u32,
    h: u32,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid & 31;
        let warp_row = (tid / 32) * (16 * RF as u32);
        let batch = thread::blockIdx_z();
        let head = thread::blockIdx_y();
        let qbase = thread::blockIdx_x() * ROWS;
        let stride = h * DU;
        let qb = (batch as usize * sq as usize + qbase as usize) * stride as usize
            + head as usize * DU as usize;
        let kb = batch as usize * sk as usize * stride as usize + head as usize * DU as usize;
        // Rebuild the generic address; helpers take generic pointers.
        let qs = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let ks = qs.add((ROWS * DU) as usize);
        let vs = ks.add((KEYS * DU) as usize);
        let qa = shared_address(qs);
        let ka = shared_address(ks);
        let va = shared_address(vs);
        // Q and the first K occupy disjoint buffers; one wait covers both.
        stage::<ROWS, PW, THREADS, QSTEPS>(qa, q.add(qb), stride, tid);
        stage::<KEYS, PW, THREADS, KSTEPS>(ka, k.add(kb), stride, tid);
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
        thread::sync_threads();
        stage::<KEYS, PW, THREADS, KSTEPS>(va, v.add(kb), stride, tid);
        let mut out = [[0.0f32; 4]; OF];
        let mut max = [f32::NEG_INFINITY; 4];
        let mut sum = [0.0f32; 4];
        let scale_log2 = scale * LOG2_E;
        // Per-lane ldmatrix addresses. Tile offsets are multiples of 8 rows
        // and PW columns, so they never touch the swizzle: every address is
        // one per-lane base per 8-column chunk pair of a plane (two for PW 16,
        // four for PW 32; the rest are dead) plus a compile-time immediate.
        let q_row = warp_row + (lane & 7) + (((lane >> 3) & 1) * 8);
        let q_col = (lane >> 4) * 4;
        let k_row = (lane & 7) + (lane >> 4) * 8;
        let k_col = ((lane & 15) >> 3) * 4;
        let v_row = lane & 15;
        let v_col = (lane >> 4) * 4;
        let q_addr = [
            qa + index::<ROWS, PW>(q_row, q_col) * 4,
            qa + index::<ROWS, PW>(q_row, q_col + 8) * 4,
            qa + index::<ROWS, PW>(q_row, q_col + 16) * 4,
            qa + index::<ROWS, PW>(q_row, q_col + 24) * 4,
        ];
        let k_addr = [
            ka + index::<KEYS, PW>(k_row, k_col) * 4,
            ka + index::<KEYS, PW>(k_row, k_col + 8) * 4,
            ka + index::<KEYS, PW>(k_row, k_col + 16) * 4,
            ka + index::<KEYS, PW>(k_row, k_col + 24) * 4,
        ];
        let v_addr = [
            va + index::<KEYS, PW>(v_row, v_col) * 4,
            va + index::<KEYS, PW>(v_row, v_col + 8) * 4,
            va + index::<KEYS, PW>(v_row, v_col + 16) * 4,
            va + index::<KEYS, PW>(v_row, v_col + 24) * 4,
        ];
        // Chunk pairs per plane.
        let cp = PW / 8;
        let mut j = 0;
        while j < sk {
            let mut s = [[0.0f32; 4]; SF];
            let mut kc = 0u32;
            while kc < KC {
                thread::__unroll_config::<0>();
                let mut a = [[0u32; 4]; RF];
                let mut rf = 0usize;
                while rf < RF {
                    thread::__unroll_config::<0>();
                    a[rf] = ldmatrix_x4_shared(
                        q_addr[(kc % cp) as usize]
                            + ((kc / cp) * ROWS * PW + rf as u32 * 16 * PW) * 4,
                    );
                    rf += 1;
                }
                let mut nt = 0usize;
                while nt < NKT {
                    thread::__unroll_config::<0>();
                    let b = ldmatrix_x4_shared(
                        k_addr[(kc % cp) as usize]
                            + ((kc / cp) * KEYS * PW + nt as u32 * 8 * PW) * 4,
                    );
                    let mut rf = 0usize;
                    while rf < RF {
                        thread::__unroll_config::<0>();
                        let f = rf * NKT + nt;
                        s[f] = mma::<BF16>(s[f], a[rf], [b[0], b[1]]);
                        s[f + 1] = mma::<BF16>(s[f + 1], a[rf], [b[2], b[3]]);
                        rf += 1;
                    }
                    nt += 2;
                }
                kc += 1;
            }
            // V readiness also releases K after all QK reads.
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            let next = j + KEYS < sk;
            if next {
                stage::<KEYS, PW, THREADS, KSTEPS>(
                    ka,
                    k.add(kb + (j as usize + KEYS as usize) * stride as usize),
                    stride,
                    tid,
                );
            }
            let mut p = [[0u32; 2]; SF];
            softmax::<BF16, RF, NKT, ND, SF, OF, MODE>(
                &s, &mut p, &mut out, &mut max, &mut sum, scale_log2,
            );
            let mut kc = 0u32;
            while kc < PVK {
                thread::__unroll_config::<0>();
                let mut d = 0usize;
                while d < ND {
                    thread::__unroll_config::<0>();
                    let b = ldmatrix_x4_trans_shared(
                        v_addr[((d as u32 / 2) % cp) as usize]
                            + ((d as u32 / 2 / cp) * KEYS * PW + kc * 16 * PW) * 4,
                    );
                    let mut rf = 0usize;
                    while rf < RF {
                        thread::__unroll_config::<0>();
                        let a = [
                            p[rf * NKT + kc as usize * 2][0],
                            p[rf * NKT + kc as usize * 2][1],
                            p[rf * NKT + kc as usize * 2 + 1][0],
                            p[rf * NKT + kc as usize * 2 + 1][1],
                        ];
                        let f = rf * ND + d;
                        out[f] = mma::<BF16>(out[f], a, [b[0], b[1]]);
                        out[f + 1] = mma::<BF16>(out[f + 1], a, [b[2], b[3]]);
                        rf += 1;
                    }
                    d += 2;
                }
                kc += 1;
            }
            // Next K readiness also releases V after all PV reads.
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            if next {
                stage::<KEYS, PW, THREADS, KSTEPS>(
                    va,
                    v.add(kb + (j as usize + KEYS as usize) * stride as usize),
                    stride,
                    tid,
                );
            }
            j += KEYS;
        }
        // Q is dead: stage the normalized output there for vector stores.
        let mut rf = 0usize;
        while rf < RF {
            thread::__unroll_config::<0>();
            let mut l0 = sum[rf * 2];
            let mut l1 = sum[rf * 2 + 1];
            l0 += shuffle_xor_f32(l0, 1);
            l0 += shuffle_xor_f32(l0, 2);
            l1 += shuffle_xor_f32(l1, 1);
            l1 += shuffle_xor_f32(l1, 2);
            let inv0 = 1.0f32 / l0;
            let inv1 = 1.0f32 / l1;
            let row = warp_row + rf as u32 * 16 + (lane >> 2);
            let mut d = 0usize;
            while d < ND {
                thread::__unroll_config::<0>();
                let col = d as u32 * 4 + (lane & 3);
                let f = out[rf * ND + d];
                *qs.add(index::<ROWS, PW>(row, col) as usize) =
                    cvt::<BF16>(f[0] * inv0, f[1] * inv0);
                *qs.add(index::<ROWS, PW>(row + 8, col) as usize) =
                    cvt::<BF16>(f[2] * inv1, f[3] * inv1);
                d += 1;
            }
            if lane & 3 == 0 {
                let base = (batch as usize * h as usize + head as usize) * sq as usize
                    + qbase as usize
                    + row as usize;
                *lse.add(base) =
                    (max[rf * 2] * scale_log2 + lg2_approx(l0)) * core::f32::consts::LN_2;
                *lse.add(base + 8) =
                    (max[rf * 2 + 1] * scale_log2 + lg2_approx(l1)) * core::f32::consts::LN_2;
            }
            rf += 1;
        }
        thread::sync_threads();
        // DU / 4 16-byte chunks per row; consecutive threads write whole rows.
        let mut step = 0;
        while step < QSTEPS {
            thread::__unroll_config::<0>();
            let c = tid + step * THREADS;
            let row = c / (DU / 4);
            let col = (c % (DU / 4)) * 4;
            let src = qs.add(index::<ROWS, PW>(row, col) as usize);
            let dst = o.add(qb + row as usize * stride as usize + col as usize);
            let x0 = *src;
            let x1 = *src.add(1);
            let x2 = *src.add(2);
            let x3 = *src.add(3);
            ptx_asm!("st.global.v4.u32 [%0], {%1, %2, %3, %4};", in("l") dst, in("r") x0, in("r") x1, in("r") x2, in("r") x3, clobber("memory"));
            step += 1;
        }
    }
}
