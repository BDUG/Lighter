//! Explicit, inference-only bridge between Candle tensors and ARM F32 kernels.
use candle_core::{Device, Tensor};
use candlelighter::arm::{matmul, Kernel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    let b = Tensor::new(&[[7f32, 8.], [9., 10.], [11., 12.]], &device)?;
    let (m, k) = a.dims2()?;
    let (bk, n) = b.dims2()?;
    if k != bk {
        return Err("matrix inner dimensions differ".into());
    }
    let left = a.flatten_all()?.to_vec1::<f32>()?;
    let right = b.flatten_all()?.to_vec1::<f32>()?;
    let result = matmul(&left, &right, m, k, n, Kernel::Auto)?;
    let bridged = Tensor::from_vec(result, (m, n), &device)?;
    let reference = a.matmul(&b)?;
    let actual = bridged.flatten_all()?.to_vec1::<f32>()?;
    let expected = reference.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(x, y)| (*x - y).abs() < 0.001));
    println!("{:?}", bridged.to_vec2::<f32>()?);
    Ok(())
}
