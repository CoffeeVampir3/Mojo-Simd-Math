from simd_math import exp_simd
from std.math import exp as libm_exp
from std.os import abort
from std.sys.info import simd_width_of


comptime W = simd_width_of[DType.float32]()

# Format-relative ULP at value 1.0 (one ULP / 1.0).
# f8 mantissa bits vary: E4M3 has 3 bits, E5M2 has 2 bits — we report E4M3 here
# because it's the typical "low-precision" target in ML.
comptime F8E4M3_ULP = Float64(0.125)             # 2^-3
comptime BF16_ULP   = Float64(0.0078125)         # 2^-7
comptime F16_ULP    = Float64(0.0009765625)      # 2^-10
comptime F32_ULP    = Float64(1.1920928955078125e-07)  # 2^-23
comptime F64_ULP    = Float64(2.220446049250313e-16)   # 2^-52


def measure_max_rel_err[
    polynomial_degree: Int,
](lo: Float64, hi: Float64, n: Int) -> Float64:
    var max_err = Float64(0.0)
    var step = (hi - lo) / Float64(n)
    var i = 0
    while i + W <= n:
        var xv = SIMD[DType.float32, W]()
        var k = 0
        while k < W:
            xv[k] = Float32(lo + Float64(i + k) * step)
            k += 1
        var got = exp_simd[polynomial_degree, W](xv)
        var j = 0
        while j < W:
            var truth = libm_exp(Float64(xv[j]))
            if truth > 0.0:
                var rel = abs(Float64(got[j]) - truth) / truth
                if rel > max_err: max_err = rel
            j += 1
        i += W
    return max_err


def report(label: String, rel: Float64):
    print(
        "  ", label, "  rel=", rel,
        "  bf16=", rel / BF16_ULP,
        "  f16=", rel / F16_ULP,
        "  f32=", rel / F32_ULP,
        sep="",
    )


def expect_le(mut failed: Int, label: String, value: Float64, bound: Float64):
    if value > bound:
        print("  FAIL ", label, ": ", value, " > ", bound, sep="")
        failed += 1
    else:
        print("  pass ", label, ": ", value, " ≤ ", bound, sep="")


def main():
    print("exp_simd error profile — sweep x ∈ [-10, 10], 2^18 samples, default flags")
    print()
    var lo = Float64(-10.0)
    var hi = Float64(10.0)
    var n = 1 << 18

    var e2 = measure_max_rel_err[2](lo, hi, n)
    var e3 = measure_max_rel_err[3](lo, hi, n)
    var e4 = measure_max_rel_err[4](lo, hi, n)
    var e5 = measure_max_rel_err[5](lo, hi, n)
    var e6 = measure_max_rel_err[6](lo, hi, n)
    var e7 = measure_max_rel_err[7](lo, hi, n)

    report(String("N=2"), e2)
    report(String("N=3"), e3)
    report(String("N=4"), e4)
    report(String("N=5"), e5)
    report(String("N=6"), e6)
    report(String("N=7"), e7)

    print()
    print("Assertions (documented bounds in exponential.mojo):")
    var failed = 0
    # N=2: documented 0.31 bf16 ULPs → bound 0.5 bf16 ULPs (60% headroom)
    expect_le(failed, String("N=2 < 0.5 bf16 ULPs (loss-free for bf16)"), e2, BF16_ULP * 0.5)
    # N=3: documented 0.45 f16 ULPs → bound 0.6 f16 ULPs
    expect_le(failed, String("N=3 < 0.6 f16 ULPs (loss-free for f16)"), e3, F16_ULP * 0.6)
    # N=4: documented ~30 f32 ULPs → bound 50
    expect_le(failed, String("N=4 < 50 f32 ULPs"), e4, F32_ULP * 50.0)
    # N=5: documented ~4.4 f32 ULPs → bound 6
    expect_le(failed, String("N=5 < 6 f32 ULPs"), e5, F32_ULP * 6.0)
    # N=6: documented ~0.9 f32 ULPs → bound 1.5 (loss-free for f32)
    expect_le(failed, String("N=6 < 1.5 f32 ULPs (loss-free for f32)"), e6, F32_ULP * 1.5)
    # N=7: documented ~0.7 f32 ULPs → bound 1.5
    expect_le(failed, String("N=7 < 1.5 f32 ULPs (saturated)"), e7, F32_ULP * 1.5)

    print()
    if failed > 0:
        print("FAILED:", failed, "assertion(s)")
        abort()
    print("All exp_simd assertions passed.")
