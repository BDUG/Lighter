//! ARM CPU kernels: runtime-dispatched NEON, optional SVE and SME F32 outer products.
//! Explicit requests fail when unavailable; `Auto` falls back to scalar kernels.
use crate::native::{NativeError, NativeResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Kernel {
    Auto,
    Scalar,
    Neon,
    Sve,
    Sme,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ArmCapabilities {
    pub neon: bool,
    pub sve: bool,
    pub sme: bool,
    pub sme_f32f32: bool,
    pub sve_compiled: bool,
    pub sme_compiled: bool,
}
impl ArmCapabilities {
    pub fn detect() -> Self {
        #[allow(unused_mut)]
        let mut caps = Self {
            sve_compiled: cfg!(all(
                target_arch = "aarch64",
                target_os = "linux",
                feature = "arm-sve"
            )),
            sme_compiled: cfg!(all(
                target_arch = "aarch64",
                target_os = "linux",
                feature = "arm-sme"
            )),
            ..Default::default()
        };
        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = std::arch::is_aarch64_feature_detected!("neon");
        }
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        {
            unsafe extern "C" {
                fn getauxval(kind: usize) -> usize;
            }
            // Linux HWCAP/HWCAP2 expose features enabled by the OS, not just CPU branding.
            let hw = unsafe { getauxval(16) };
            let hw2 = unsafe { getauxval(26) };
            caps.sve = hw & (1 << 22) != 0;
            caps.sme = hw2 & (1 << 23) != 0;
            caps.sme_f32f32 = hw2 & (1 << 29) != 0;
        }
        // Other operating systems use NEON here; no unsupported SVE/SME probing.
        caps
    }
    pub fn available(self, kernel: Kernel) -> bool {
        match kernel {
            Kernel::Auto | Kernel::Scalar => true,
            Kernel::Neon => self.neon,
            Kernel::Sve => self.sve && self.sve_compiled,
            Kernel::Sme => self.sme && self.sme_f32f32 && self.sme_compiled,
        }
    }
    pub fn vector_kernel(self) -> Kernel {
        if self.available(Kernel::Sve) {
            Kernel::Sve
        } else if self.neon {
            Kernel::Neon
        } else {
            Kernel::Scalar
        }
    }
}

pub fn dot(a: &[f32], b: &[f32], kernel: Kernel) -> NativeResult<f32> {
    if a.len() != b.len() {
        return Err(NativeError("dot-product lengths differ".into()));
    }
    let caps = ArmCapabilities::detect();
    let selected = if kernel == Kernel::Auto {
        caps.vector_kernel()
    } else {
        kernel
    };
    if selected == Kernel::Sme {
        return Err(NativeError("SME is a matrix kernel; use matmul".into()));
    }
    if !caps.available(selected) {
        return Err(NativeError(format!(
            "requested ARM kernel {selected:?} is unavailable"
        )));
    }
    Ok(dot_selected(a, b, selected))
}
/// Internal prevalidated path used by native F32 matrix projections.
pub(crate) fn dot_auto(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    static KERNEL: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();
    dot_selected(
        a,
        b,
        *KERNEL.get_or_init(|| ArmCapabilities::detect().vector_kernel()),
    )
}
fn dot_selected(a: &[f32], b: &[f32], selected: Kernel) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if selected == Kernel::Neon {
        // Runtime dispatch above establishes NEON support; slice lengths were checked.
        return unsafe { neon_dot(a, b) };
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sve"))]
    if selected == Kernel::Sve {
        // Linux HWCAP verifies SVE is usable by this process before entering assembly.
        return unsafe { lighter_arm_sve_dot(a.as_ptr(), b.as_ptr(), a.len()) };
    }
    let _ = selected;
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_f32(0.0);
    let end = a.len() / 4 * 4;
    for offset in (0..end).step_by(4) {
        // Each load stays inside both checked slices. Unaligned loads are supported.
        sum = vaddq_f32(
            sum,
            vmulq_f32(
                vld1q_f32(a.as_ptr().add(offset)),
                vld1q_f32(b.as_ptr().add(offset)),
            ),
        );
    }
    vaddvq_f32(sum)
        + a[end..]
            .iter()
            .zip(&b[end..])
            .map(|(x, y)| x * y)
            .sum::<f32>()
}

/// Row-major A[m,k] * B[k,n]. Auto uses SME when compiled and OS-enabled,
/// otherwise vectorized dot products. Explicit kernels never silently fall back.
pub fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    kernel: Kernel,
) -> NativeResult<Vec<f32>> {
    if m.checked_mul(k) != Some(a.len()) || k.checked_mul(n) != Some(b.len()) {
        return Err(NativeError(
            "matrix dimensions differ from buffer lengths".into(),
        ));
    }
    let length = m
        .checked_mul(n)
        .ok_or_else(|| NativeError("matrix output dimensions overflow".into()))?;
    let caps = ArmCapabilities::detect();
    let selected = if kernel == Kernel::Auto {
        if caps.available(Kernel::Sme) {
            Kernel::Sme
        } else {
            caps.vector_kernel()
        }
    } else {
        kernel
    };
    if !caps.available(selected) {
        return Err(NativeError(format!(
            "requested ARM kernel {selected:?} is unavailable"
        )));
    }
    if length == 0 || k == 0 {
        return Ok(vec![0.; length]);
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sme"))]
    if selected == Kernel::Sme {
        return sme_matmul(a, b, m, k, n);
    }
    let mut transposed = vec![0.; b.len()];
    for row in 0..k {
        for column in 0..n {
            transposed[column * k + row] = b[row * n + column];
        }
    }
    Ok((0..length)
        .into_par_iter()
        .map(|index| {
            dot_selected(
                &a[index / n * k..(index / n + 1) * k],
                &transposed[index % n * k..(index % n + 1) * k],
                selected,
            )
        })
        .collect())
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sve"))]
core::arch::global_asm!(
    r#"
.arch_extension sve
.text
.global lighter_arm_sve_dot
.type lighter_arm_sve_dot,%function
lighter_arm_sve_dot:
    mov x3, #0
    dup z0.s, #0
1:
    whilelo p0.s, x3, x2
    b.none 2f
    ld1w z1.s, p0/z, [x0, x3, lsl #2]
    ld1w z2.s, p0/z, [x1, x3, lsl #2]
    fmla z0.s, p0/m, z1.s, z2.s
    incw x3
    b 1b
2:
    ptrue p0.s
    faddv s0, p0, z0.s
    ret
.size lighter_arm_sve_dot, .-lighter_arm_sve_dot
"#,
    options(raw)
);
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sve"))]
unsafe extern "C" {
    fn lighter_arm_sve_dot(a: *const f32, b: *const f32, len: usize) -> f32;
}

#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sme"))]
core::arch::global_asm!(
    r#"
.arch_extension sme
.text
.global lighter_arm_sme_lanes
.type lighter_arm_sme_lanes,%function
lighter_arm_sme_lanes:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    cntw x0
    smstop
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
.size lighter_arm_sme_lanes, .-lighter_arm_sme_lanes
.global lighter_arm_sme_tile
.type lighter_arm_sme_tile,%function
lighter_arm_sme_tile:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    smstart
    zero {za}
    ptrue p0.s
    cntb x9
    cntw x10
    cmp x10, x4
    b.ne 3f
1:
    ld1w z0.s, p0/z, [x0]
    ld1w z1.s, p0/z, [x1]
    fmopa za0.s, p0/m, p0/m, z0.s, z1.s
    add x0, x0, x9
    add x1, x1, x9
    subs x3, x3, #1
    b.ne 1b
    mov w12, #0
2:
    st1w {za0h.s[w12, 0]}, p0, [x2]
    add x2, x2, x9
    add w12, w12, #1
    cmp x12, x10
    b.lo 2b
    mov x0, #0
    b 4f
3:
    mov x0, #1
4:
    smstop
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
.size lighter_arm_sme_tile, .-lighter_arm_sme_tile
"#,
    options(raw)
);
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sme"))]
unsafe extern "C" {
    fn lighter_arm_sme_lanes() -> usize;
    fn lighter_arm_sme_tile(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        k: usize,
        lanes: usize,
    ) -> usize;
}
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "arm-sme"))]
fn sme_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> NativeResult<Vec<f32>> {
    // This kernel enters private streaming/ZA state and restores the base AAPCS
    // callee-saved SIMD registers. It must be called from ordinary non-streaming Rust.
    let lanes = unsafe { lighter_arm_sme_lanes() };
    if !(4..=64).contains(&lanes) {
        return Err(NativeError("unexpected SME streaming vector length".into()));
    }
    let packed_len = k
        .checked_mul(lanes)
        .ok_or_else(|| NativeError("SME packing size overflow".into()))?;
    let mut output = vec![0.; m * n];
    let mut left = vec![0.; packed_len];
    let mut right = vec![0.; packed_len];
    let mut tile = vec![0.; lanes * lanes];
    for row in (0..m).step_by(lanes) {
        for column in (0..n).step_by(lanes) {
            left.fill(0.);
            right.fill(0.);
            for depth in 0..k {
                for lane in 0..lanes {
                    if row + lane < m {
                        left[depth * lanes + lane] = a[(row + lane) * k + depth];
                    }
                    if column + lane < n {
                        right[depth * lanes + lane] = b[depth * n + column + lane];
                    }
                }
            }
            // Buffers contain exactly k streaming vectors; k>0 was checked by matmul.
            if unsafe {
                lighter_arm_sme_tile(left.as_ptr(), right.as_ptr(), tile.as_mut_ptr(), k, lanes)
            } != 0
            {
                return Err(NativeError(
                    "SME vector length changed during packing".into(),
                ));
            }
            for i in 0..lanes.min(m - row) {
                for j in 0..lanes.min(n - column) {
                    output[(row + i) * n + column + j] = tile[i * lanes + j];
                }
            }
        }
    }
    Ok(output)
}
