#![cfg(feature = "native")]
use candlelighter::arm::{dot, matmul, ArmCapabilities, Kernel};
#[test]
fn kernels_match_scalar_for_tails_and_tiles() {
    let caps = ArmCapabilities::detect();
    for kernel in [Kernel::Scalar, Kernel::Auto, Kernel::Neon, Kernel::Sve] {
        if !caps.available(kernel) {
            assert!(dot(&[1.], &[1.], kernel).is_err());
            continue;
        }
        for len in [0, 1, 3, 4, 7, 16, 65, 137] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.3).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();
            let expected = dot(&a, &b, Kernel::Scalar).unwrap();
            assert!(
                (dot(&a, &b, kernel).unwrap() - expected).abs() < 0.001,
                "{kernel:?} len={len}"
            );
        }
    }
    for (m, k, n) in [(1, 1, 1), (3, 7, 5), (17, 19, 21), (0, 3, 4), (2, 0, 4)] {
        let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.3).sin()).collect();
        let b: Vec<_> = (0..k * n).map(|i| (i as f32 * 0.2).cos()).collect();
        let expected = matmul(&a, &b, m, k, n, Kernel::Scalar).unwrap();
        for kernel in [Kernel::Auto, Kernel::Neon, Kernel::Sve, Kernel::Sme] {
            if !caps.available(kernel) {
                assert!(matmul(&a, &b, m, k, n, kernel).is_err());
                continue;
            }
            let actual = matmul(&a, &b, m, k, n, kernel).unwrap();
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 0.001),
                "{kernel:?}"
            );
        }
    }
}
#[test]
fn malformed_shapes_and_explicit_sme_dot_fail() {
    assert!(dot(&[1.], &[], Kernel::Auto).is_err());
    assert!(dot(&[], &[], Kernel::Sme).is_err());
    assert!(matmul(&[], &[], usize::MAX, 2, 1, Kernel::Auto).is_err());
    assert!(matmul(&[], &[], usize::MAX, 0, 2, Kernel::Auto).is_err());
    assert!(matmul(&[1.], &[], 1, 1, 1, Kernel::Auto).is_err());
}
