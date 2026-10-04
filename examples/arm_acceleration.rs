//! No weights or vendor SDK required. Runtime dispatch checks OS-enabled features.
use candlelighter::arm::{dot, matmul, ArmCapabilities, Kernel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&ArmCapabilities::detect())?
    );
    let a = vec![1., 2., 3., 4., 5.];
    let b = vec![5., 4., 3., 2., 1.];
    assert!((dot(&a, &b, Kernel::Auto)? - 35.).abs() < 0.001);
    let product = matmul(
        &[1., 2., 3., 4., 5., 6.],
        &[7., 8., 9., 10., 11., 12.],
        2,
        3,
        2,
        Kernel::Auto,
    )?;
    assert_eq!(product, vec![58., 64., 139., 154.]);
    println!("dot=35; matrix product={product:?}");
    // Explicit requests report an error rather than silently running another kernel.
    for kernel in [Kernel::Neon, Kernel::Sve, Kernel::Sme] {
        match matmul(&a, &b, 1, 5, 1, kernel) {
            Ok(value) => println!("{kernel:?}: {value:?}"),
            Err(error) => println!("{kernel:?}: {error}"),
        }
    }
    Ok(())
}
