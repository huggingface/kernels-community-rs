//! Device kernels.
//!
//! The kernels (except [`kernels::fatrelu_and_mul`]) are generic over the
//! activation, which is selected by the `ACT` const generic (one of the
//! activation constants).
//!
//! - [`kernels::activation`] gets the input and output as flat `[numel]`
//!   tensors, one tile of `S` elements per program.
//! - The gated kernels get the input as `[tokens, 2, d]`, so the two halves
//!   of a row are `[row, 0, ..]` and `[row, 1, ..]`, and the output as
//!   `[tokens, d]`. One program handles `B` elements of one token. Loads are
//!   zero-padded and stores are clipped, so `d` need not be a multiple of `B`.
//!


// NOTE: cuTile kernels cannot be generic over a trait like the cuda-oxide,
// but the JIT only compiles the selected branch on const matching, so that
// seemed like the best solution.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    // Values of the `ACT` const generic.
    pub const SILU: i32 = 0;
    pub const GELU: i32 = 1;
    pub const GELU_TANH: i32 = 2;
    pub const GELU_NEW: i32 = 3;
    pub const GELU_FAST: i32 = 4;
    pub const GELU_QUICK: i32 = 5;

    fn fma_f32<const S: [i32; 1]>(
        a: Tile<f32, S>,
        b: Tile<f32, S>,
        c: Tile<f32, S>,
    ) -> Tile<f32, S> {
        fma(a, b, c, rounding::NearestEven, ftz::Disabled)
    }

    /// Round an f32 tile to `E` and widen it again. The CUDA kernels round
    /// intermediate results to the element type in some activations.
    fn round_to<E: ElementType, const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let narrow: Tile<E, S> = convert_tile(x);
        convert_tile(narrow)
    }

    /// `erf` approximation for f32
    fn erf_f32<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        // With `a = |x|`:
        //
        // - `a < 1.125`: `erf(x) = x + x * p(x^2 - 0.5)`.
        // - `a >= 1.125`: `erf(x) = copysign(1 - 2^-|r|, x)` with
        //   `r = x + x * q(a - 2)`, where `|r|` approximates `-log2(erfc(a))`.
        //   f32 `erf` is 1 from about 3.92 on, where `2^-|r|` is negligible.
        //
        // `p` and `q` have the same degree, so this is a single Horner chain
        // with per-element coefficients, plus one `exp2` (a single instruction,
        // unlike `exp`). The coefficients are fitted by `scripts/fit_erf.py`,
        // which also checks the error for all f32 (at most 0.95 ulp).

        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        let split: Tile<f32, S> = constant(1.125f32, x.shape());
        let small_center: Tile<f32, S> = constant(-0.5f32, x.shape());
        let large_center: Tile<f32, S> = constant(2.0f32, x.shape());
        let a: Tile<f32, S> = absf(x);
        let big: Tile<bool, S> = ge_tile(a, split);
        let z: Tile<f32, S> = select(big, a - large_center, fma_f32(x, x, small_center));

        // Coefficients of `p` and `q`, constant term first.
        let p0: Tile<f32, S> = constant(-0.034531254f32, x.shape());
        let p1: Tile<f32, S> = constant(-0.28107217f32, x.shape());
        let p2: Tile<f32, S> = constant(0.07940956f32, x.shape());
        let p3: Tile<f32, S> = constant(-0.018283825f32, x.shape());
        let p4: Tile<f32, S> = constant(0.0034841618f32, x.shape());
        let p5: Tile<f32, S> = constant(-0.0005638319f32, x.shape());
        let p6: Tile<f32, S> = constant(6.934507e-05f32, x.shape());
        let q0: Tile<f32, S> = constant(2.869987f32, x.shape());
        let q1: Tile<f32, S> = constant(1.2520355f32, x.shape());
        let q2: Tile<f32, S> = constant(0.040328786f32, x.shape());
        let q3: Tile<f32, S> = constant(-0.008280593f32, x.shape());
        let q4: Tile<f32, S> = constant(0.0015708113f32, x.shape());
        let q5: Tile<f32, S> = constant(-0.00026514402f32, x.shape());
        let q6: Tile<f32, S> = constant(1.9896934e-05f32, x.shape());
        let c0: Tile<f32, S> = select(big, q0, p0);
        let c1: Tile<f32, S> = select(big, q1, p1);
        let c2: Tile<f32, S> = select(big, q2, p2);
        let c3: Tile<f32, S> = select(big, q3, p3);
        let c4: Tile<f32, S> = select(big, q4, p4);
        let c5: Tile<f32, S> = select(big, q5, p5);
        let c6: Tile<f32, S> = select(big, q6, p6);
        let h: Tile<f32, S> = fma_f32(c6, z, c5);
        let h: Tile<f32, S> = fma_f32(h, z, c4);
        let h: Tile<f32, S> = fma_f32(h, z, c3);
        let h: Tile<f32, S> = fma_f32(h, z, c2);
        let h: Tile<f32, S> = fma_f32(h, z, c1);
        let h: Tile<f32, S> = fma_f32(h, z, c0);
        let r: Tile<f32, S> = fma_f32(h, x, x);

        // Where 2^-|r| is denormal, 1 - 2^-|r| rounds to 1 anyway.
        let tail: Tile<f32, S> = one - exp2(negf(absf(r)), ftz::Enabled);
        let tail: Tile<f32, S> = select(lt_tile(x, zero), negf(tail), tail);
        select(big, tail, r)
    }

    /// SiLU, `x * sigmoid(x)`.
    fn silu_f32<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        x / (one + exp(negf(x)))
    }

    /// GELU, `x * cdf(x)`, exact (PyTorch `approximate="none"`).
    fn gelu_f32<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let half: Tile<f32, S> = constant(0.5f32, x.shape());
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        // 1 / sqrt(2), rounded to f32.
        let alpha: Tile<f32, S> = constant(0.707_106_77f32, x.shape());
        x * half * (one + erf_f32(x * alpha))
    }

    /// GELU with the tanh approximation (PyTorch `approximate="tanh"`).
    fn gelu_tanh_f32<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let half: Tile<f32, S> = constant(0.5f32, x.shape());
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        // sqrt(2) * (2 / sqrt(pi)) * 0.5, rounded to f32.
        let beta: Tile<f32, S> = constant(0.797_884_56f32, x.shape());
        let kappa: Tile<f32, S> = constant(0.044715f32, x.shape());
        let x_cube: Tile<f32, S> = x * x * x;
        // Like nvcc, contract `x + kappa * x_cube` to an FMA.
        let inner: Tile<f32, S> = beta * fma_f32(kappa, x_cube, x);
        half * x * (one + tanh(inner))
    }

    /// `x * sigmoid(1.702 * x)`.
    fn gelu_quick_f32<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        let alpha: Tile<f32, S> = constant(1.702f32, x.shape());
        x / (one + exp(negf(alpha * x)))
    }

    /// The GELU approximation used by GPT-2. Like the CUDA kernel, which
    /// computes in the element type, results are rounded to `E` after each
    /// operation.
    fn gelu_new_f32<E: ElementType, const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let half: Tile<f32, S> = constant(0.5f32, x.shape());
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        let kappa: Tile<f32, S> = constant(0.044715f32, x.shape());
        let beta: Tile<f32, S> = constant(0.797_884_56f32, x.shape());
        let x2: Tile<f32, S> = round_to::<E, S>(x * x);
        let x3: Tile<f32, S> = round_to::<E, S>(x2 * x);
        let kx3: Tile<f32, S> = round_to::<E, S>(kappa * x3);
        let inner: Tile<f32, S> = round_to::<E, S>(x + kx3);
        let scaled: Tile<f32, S> = round_to::<E, S>(beta * inner);
        let t: Tile<f32, S> = round_to::<E, S>(tanh(scaled));
        let half_x: Tile<f32, S> = round_to::<E, S>(half * x);
        let one_t: Tile<f32, S> = round_to::<E, S>(one + t);
        half_x * one_t
    }

    /// A faster tanh-based GELU approximation, rounding to `E` like
    /// [`gelu_new_f32`].
    fn gelu_fast_f32<E: ElementType, const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let half: Tile<f32, S> = constant(0.5f32, x.shape());
        let one: Tile<f32, S> = constant(1.0f32, x.shape());
        let kappa: Tile<f32, S> = constant(0.044715f32, x.shape());
        let beta: Tile<f32, S> = constant(0.797_884_56f32, x.shape());
        let scaled: Tile<f32, S> = round_to::<E, S>(x * beta);
        let kx: Tile<f32, S> = round_to::<E, S>(kappa * x);
        let kx2: Tile<f32, S> = round_to::<E, S>(kx * x);
        let cubic: Tile<f32, S> = round_to::<E, S>(one + kx2);
        let inner: Tile<f32, S> = round_to::<E, S>(scaled * cubic);
        let t: Tile<f32, S> = round_to::<E, S>(tanh(inner));
        let half_x: Tile<f32, S> = round_to::<E, S>(half * x);
        let one_t: Tile<f32, S> = round_to::<E, S>(one + t);
        half_x * one_t
    }

    /// FATReLU: `x` above `threshold`, zero otherwise.
    fn fatrelu_f32<const S: [i32; 1]>(x: Tile<f32, S>, threshold: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        select(gt_tile(x, threshold), x, zero)
    }

    /// Apply the activation selected by `ACT`. Activations that compute in
    /// the element type round to `E`.
    fn apply<E: ElementType, const ACT: i32, const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        if ACT == SILU {
            silu_f32(x)
        } else if ACT == GELU {
            gelu_f32(x)
        } else if ACT == GELU_TANH {
            gelu_tanh_f32(x)
        } else if ACT == GELU_NEW {
            gelu_new_f32::<E, S>(x)
        } else if ACT == GELU_FAST {
            gelu_fast_f32::<E, S>(x)
        } else {
            gelu_quick_f32(x)
        }
    }

    /// Element-wise activation: `out[i] = act(input[i])`.
    #[cutile::entry()]
    pub fn activation<E: ElementType, const S: [i32; 1], const ACT: i32>(
        out: &mut Tensor<E, S>,
        input: &Tensor<E, { [-1] }>,
    ) {
        let x: Tile<f32, S> = convert_tile(input.load_like(out));
        let y: Tile<f32, S> = apply::<E, ACT, S>(x);
        let y: Tile<E, S> = convert_tile(y);
        out.store(y);
    }

    /// Load the two halves of the `B` elements of this program.
    fn load_halves<E: ElementType, const B: i32>(
        input: &Tensor<E, { [-1, -1, -1] }>,
    ) -> (Tile<f32, { [B] }>, Tile<f32, { [B] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let token = pid.0;
        let col = pid.1;
        let x: Tile<E, { [1, 1, B] }> = input.load_tile(shape![1, 1, B], [token, 0i32, col]);
        let y: Tile<E, { [1, 1, B] }> = input.load_tile(shape![1, 1, B], [token, 1i32, col]);
        let x: Tile<f32, { [B] }> = convert_tile(x.reshape(shape![B]));
        let y: Tile<f32, { [B] }> = convert_tile(y.reshape(shape![B]));
        (x, y)
    }

    /// Gated activation: `out[t, i] = act(x) * y` with `x = input[t, 0, i]`
    /// and `y = input[t, 1, i]`, or `x * act(y)` when `ACT_FIRST` is false.
    /// Like the CUDA kernel, `act` is rounded to `E` before the product.
    #[cutile::entry()]
    pub fn act_and_mul<E: ElementType, const B: i32, const ACT: i32, const ACT_FIRST: bool>(
        out: &mut Tensor<E, { [1, B] }>,
        input: &Tensor<E, { [-1, -1, -1] }>,
    ) {
        let halves: (Tile<f32, { [B] }>, Tile<f32, { [B] }>) = load_halves::<E, B>(input);
        let x: Tile<f32, { [B] }> = halves.0;
        let y: Tile<f32, { [B] }> = halves.1;
        let z: Tile<f32, { [B] }> = if ACT_FIRST {
            let act: Tile<f32, { [B] }> = apply::<E, ACT, { [B] }>(x);
            let act: Tile<f32, { [B] }> = round_to::<E, { [B] }>(act);
            act * y
        } else {
            let act: Tile<f32, { [B] }> = apply::<E, ACT, { [B] }>(y);
            let act: Tile<f32, { [B] }> = round_to::<E, { [B] }>(act);
            x * act
        };
        let z: Tile<E, { [B] }> = convert_tile(z);
        out.store(z.reshape(shape![1, B]));
    }

    /// `out[t, i] = fatrelu(input[t, 0, i], threshold) * input[t, 1, i]`.
    #[cutile::entry()]
    pub fn fatrelu_and_mul<E: ElementType, const B: i32>(
        out: &mut Tensor<E, { [1, B] }>,
        input: &Tensor<E, { [-1, -1, -1] }>,
        threshold: f32,
    ) {
        let halves: (Tile<f32, { [B] }>, Tile<f32, { [B] }>) = load_halves::<E, B>(input);
        let x: Tile<f32, { [B] }> = halves.0;
        let y: Tile<f32, { [B] }> = halves.1;
        let threshold: Tile<f32, { [B] }> = threshold.broadcast(shape![B]);
        let act: Tile<f32, { [B] }> = fatrelu_f32(x, threshold);
        let z: Tile<E, { [B] }> = convert_tile(act * y);
        out.store(z.reshape(shape![1, B]));
    }
}
