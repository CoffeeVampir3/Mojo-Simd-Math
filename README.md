# Mojo SIMD Math

`simd_math` is a small Mojo package containing SIMD-oriented math routines and
support utilities. The transcendental routines are finite-input approximations
intended for numerical and ML workloads where predictable vector throughput is
important. They are measured against standard library math functions over fixed
test sweeps, with assertions in `simd_math/tests`.

The approximation routines are not full IEEE 754 or libm replacements. Some
functions expose optional correction paths for inexpensive edge improvements,
but special values, exception flags, exact rounding, and subnormal behavior are
not generally covered. The `fast_flags` module can enable FTZ/DAZ subnormal
zeroing, which is the intended use case for these primitives in general.

## Contents

| Area | Operation | Input/output type | File | Summary |
| --- | --- | --- | --- | --- |
| Exponential | `exp_simd` | `SIMD[float32, width]` | `simd_math/exponential.mojo` | Polynomial approximation to `exp(x)` with configurable degree 2-7. |
| Logarithm | `log_simd` | `SIMD[float32, width]` | `simd_math/logarithm.mojo` | Polynomial approximation to `log(x)` with configurable degree 2-7. |
| Trigonometry | `sincos_simd` | `SIMD[float64, width]` | `simd_math/sincos.mojo` | Joint sine/cosine approximation with configurable degree 4-10 and optional Cody-Waite range reduction. |
| Matrix/SIMD utilities | transpose, interleave, reductions, top-k helpers | generic SIMD/integer types | `simd_math/matrixops.mojo` | Compile-time SIMD layout and reduction utilities. |
| Runtime flags | `set_subnormal_zeroing`, `get_mxcsr` | MXCSR control | `simd_math/fast_flags.mojo` | Enables or reads x86 SIMD floating-point control flags. |

## Measurement Notes

The tables below report the values measured by the repository tests. The ULP
columns are normalized by the format ULP at value 1.0, matching the test code;
they are useful as a relative scale but are not a full per-result ULP-distance
analysis.

All sweeps use `2^18` samples. The tests compare against standard library
`exp`, `log`, `sin`, and `cos` for validation and fail if documented bounds are
exceeded.

Run the validation suite with:

```bash
pixi run test
```

Individual tests are also available:

```bash
pixi run test-exp
pixi run test-log
pixi run test-sincos
```

## `exp_simd`

Measured over `x` in `[-10, 10]`, using max relative error for `float32`
inputs and outputs.

| Polynomial degree | Max relative error | bf16 ULPs | f16 ULPs | f32 ULPs | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | `2.326e-03` | `2.977e-01` | `2.382e+00` | `1.951e+04` | Coarse approximation; below 0.5 bf16 ULP on this sweep. |
| 3 | `4.315e-04` | `5.524e-02` | `4.419e-01` | `3.620e+03` | Below 0.6 f16 ULP on this sweep. |
| 4 | `3.641e-06` | `4.661e-04` | `3.729e-03` | `3.055e+01` | Sub-f16 relative error; not f32 ULP-level. |
| 5 | `5.142e-07` | `6.582e-05` | `5.266e-04` | `4.314e+00` | A few f32 ULPs by this scale. |
| 6 | `1.028e-07` | `1.316e-05` | `1.053e-04` | `8.625e-01` | Recommended f32-precision tier in the current comments. |
| 7 | `8.608e-08` | `1.102e-05` | `8.814e-05` | `7.221e-01` | Similar f32-scale behavior to degree 6. |

Relevant options:

| Option | Default | Effect |
| --- | --- | --- |
| `ieee_corrections` | `False` | Uses a lower clamp for underflow-range inputs and a split `ln(2)` range-reduction path. This is not full IEEE conformance. |
| `never_overshoot` | `False` | Selects biased coefficients intended to avoid overshoot in the approximation. |

## `log_simd`

Measured over `x` in `[0.1, 10]` with log-uniform sampling, using max relative
error for positive `float32` inputs. Samples where the reference result is zero
are skipped by the test when forming relative error.

| Polynomial degree | Max relative error | bf16 ULPs | f16 ULPs | f32 ULPs | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | `6.126e-07` | `7.842e-05` | `6.273e-04` | `5.139e+00` | Small f32-scale residual; below the asserted 8 f32 ULP bound. |
| 3 | `2.110e-07` | `2.701e-05` | `2.161e-04` | `1.770e+00` | Current recommended degree in source comments. |
| 4 | `2.110e-07` | `2.701e-05` | `2.161e-04` | `1.770e+00` | No measured improvement over degree 3 on this sweep. |
| 5 | `2.110e-07` | `2.701e-05` | `2.161e-04` | `1.770e+00` | Same measured floor as degree 3. |
| 6 | `2.110e-07` | `2.701e-05` | `2.161e-04` | `1.770e+00` | Same measured floor as degree 3. |
| 7 | `2.110e-07` | `2.701e-05` | `2.161e-04` | `1.770e+00` | Same measured floor as degree 3. |

Relevant options:

| Option | Default | Effect |
| --- | --- | --- |
| `ieee_corrections` | `False` | Clamps inputs into `[FLT_MIN_NORMAL, FLT_MAX]` before the approximation. This avoids some bit-pattern issues but does not implement IEEE `log` special-case semantics. |

## `sincos_simd`

Measured over `theta` in `[-pi, pi]`, using max absolute error for `float64`
inputs and outputs. Since sine and cosine are bounded by 1, this is also a
useful relative scale near peak magnitude.

| Polynomial degree | Max sin abs error | Max cos abs error | Worst f32 ULPs | Worst f64 ULPs | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | `1.184e-06` | `1.184e-06` | `9.931e+00` | `5.332e+09` | Approximate f32-scale tier. |
| 5 | `6.693e-09` | `6.693e-09` | `5.614e-02` | `3.014e+07` | Sub-f32 by this scale. |
| 6 | `2.661e-11` | `2.661e-11` | `2.233e-04` | `1.199e+05` | Intermediate f64-scale tier. |
| 7 | `7.860e-14` | `7.860e-14` | `6.594e-07` | `3.540e+02` | Hundreds of f64 ULPs by this scale. |
| 8 | `7.008e-16` | `7.650e-16` | `6.417e-09` | `3.445e+00` | Current f64-precision tier in source comments. |
| 9 | `6.384e-16` | `6.939e-16` | `5.821e-09` | `3.125e+00` | Near the measured saturation point. |
| 10 | `6.384e-16` | `6.939e-16` | `5.821e-09` | `3.125e+00` | Similar measured behavior to degree 9. |

Large-angle range-reduction behavior is also tested for degree 8 over
`theta` in `[-1e6, 1e6]`.

| Range reduction | Max sin abs error | Max cos abs error | Worst f32 ULPs | Worst f64 ULPs | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Naive | `3.882e-11` | `3.880e-11` | `3.256e-04` | `1.748e+05` | Faster reduction path; loses precision at large angles. |
| Cody-Waite | `7.078e-16` | `8.847e-16` | `7.421e-09` | `3.984e+00` | More accurate large-angle path for the tested range. |

Relevant options:

| Option | Default | Effect |
| --- | --- | --- |
| `cody_waite` | `False` | Uses a two-part `2*pi` split during range reduction. This improves large finite angles in the documented range, but is not arbitrary-precision libm range reduction. |
