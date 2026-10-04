#!/usr/bin/env python3
"""Cross-compile ARM kernel bodies/assembly without requiring a cross C toolchain.
Serde derives and Rayon iteration are replaced only in a temporary harness; this
check validates kernel code generation, not the complete crate or hardware execution.
Requires: rustup target add aarch64-unknown-linux-gnu
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "lib/arm.rs").read_text()
source = source.replace("use rayon::prelude::*;", "")
source = source.replace("use serde::{Deserialize, Serialize};", "")
source = source.replace("Serialize, Deserialize,", "")
source = source.replace(".into_par_iter()", ".into_iter()")
with tempfile.TemporaryDirectory(prefix="lighter-arm-codegen-") as directory:
    directory = Path(directory)
    (directory / "arm.rs").write_text(source)
    (directory / "check.rs").write_text(
        "mod native { #[derive(Debug)] pub struct NativeError(pub String); "
        "pub type NativeResult<T> = Result<T, NativeError>; }\n"
        '#[path="arm.rs"] pub mod arm;\n'
    )
    subprocess.run([
        "rustc", "--edition", "2021", "--cfg", 'feature="arm-sve"',
        "--cfg", 'feature="arm-sme"', "--target", "aarch64-unknown-linux-gnu",
        "--crate-type", "lib", "--emit", "obj", "-A", "dead_code",
        str(directory / "check.rs"), "-o", str(directory / "arm.o"),
    ], check=True)
print("AArch64 NEON/SVE/SME kernel code generation passed (hardware not executed).")
