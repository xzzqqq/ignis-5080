//! Build and link the CUDA leaf on Linux. CPU builds need no CUDA toolchain.
//! Every source change reruns the build; an existing archive never bypasses it.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The kernel source trees whose contents force a leaf rebuild, relative to
/// `kernel/`. A whitelist rather than a walk of `kernel/` itself: the build
/// directory lives there too, and asking cargo to watch a file cmake writes
/// would rebuild on every single invocation, forever.
const KERNEL_SOURCE_ROOTS: &[&str] = &["src", "include", "tests", "vendor/src", "vendor/include"];

/// Files directly under `kernel/` that decide how the leaf is built. Both
/// host scripts are watched on both hosts: which one runs depends on the
/// target, but an edit to either is an edit to how this leaf is built.
const KERNEL_BUILD_FILES: &[&str] = &["CMakeLists.txt", "build.sh"];

/// Emit a `rerun-if-changed` for every file under `dir`, recursively. A
/// missing or unreadable directory is silently skipped: the whitelist above
/// is a superset (`tests` is optional), and a directory that does not exist
/// cannot be the thing that went stale.
fn watch_tree(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            watch_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    // The `cuda` feature gates all kernel linking; the default (CPU) build is
    // pure Rust and links nothing.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernel");
    let build_dir = kernel_dir.join("build");
    // Two archives: the leaf's own C ABI surface and the vendored reference
    // substrate it is built on (ADR 0010, kernel/vendor/VENDOR.md).
    let libraries = ["ignis_kernel", "ignis_vendor"];

    // Declared before the build runs, so a build that fails still leaves
    // cargo watching the source that must change to fix it.
    for root in KERNEL_SOURCE_ROOTS {
        watch_tree(&kernel_dir.join(root));
    }
    for file in KERNEL_BUILD_FILES {
        println!("cargo:rerun-if-changed={}", kernel_dir.join(file).display());
    }

    assert_eq!(env::var("CARGO_CFG_TARGET_OS").as_deref(), Ok("linux"),
               "CUDA builds run on remote Linux; use a CPU build on the Laptop");
    let mut command = Command::new("bash");
    command.arg(kernel_dir.join("build.sh"));
    let script_name = "kernel/build.sh";
    let out = command.output().unwrap_or_else(|e| panic!("failed to run {script_name}: {e}"));
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        panic!(
            "kernel leaf build failed ({}) — check the toolchain \
             (cmake + ninja + nvcc under CUDA_HOME) and re-run",
            out.status
        );
    }
    // A build that reported success but produced no archive would otherwise
    // fail much later, as an opaque linker error.
    for name in libraries {
        let lib =
            build_dir.join(format!("lib{name}.a"));
        if !lib.exists() {
            eprintln!("{}", String::from_utf8_lossy(&out.stdout));
            panic!("kernel leaf build reported success but {} is missing", lib.display());
        }
    }

    println!("cargo:rustc-link-search={}", build_dir.display());
    for name in libraries {
        println!("cargo:rustc-link-lib=static={name}");
    }
    // The two archives depend on each other (kernel/CMakeLists.txt says why),
    // and on ELF that is an ordering problem: ld walks its inputs once, so a
    // symbol `ignis_vendor` needs from `ignis_kernel` is undefined because
    // `ignis_kernel` was already behind it on the line. Naming the first one
    // again is the ordinary fix, and the second copy contributes nothing when
    // the first already resolved everything. The MSVC linker re-scans on its
    // own and needs no such help.
    println!("cargo:rustc-link-lib=static={}", libraries[0]);

    // Only the CUDA runtime is linked, never the driver's own libraries
    // (libcuda.so.1 / nvcuda.dll, libnvidia-ml.so.1 / nvml.dll), which ship
    // with the NVIDIA driver rather than the toolkit: a binary that names one
    // does not start on a machine without the driver -- on Linux the loader
    // refuses before `main` -- not even for `--help` or `model download`
    // (GitHub #329). The leaf reaches the driver API through the runtime
    // (kernel/src/cuda_driver_shim.cu) and opens NVML when it first reads it
    // (kernel/src/device.cu).
        let cuda = env::var("CUDA_HOME")
            .or_else(|_| env::var("CUDA_PATH"))
            .unwrap_or_else(|_| "/usr/local/cuda".into());
        let cuda_lib = PathBuf::from(&cuda).join("lib64");
        if cuda_lib.exists() {
            println!("cargo:rustc-link-search=native={}", cuda_lib.display());
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
        // The leaf opens NVML with dlopen (kernel/src/shared_library.cu),
        // which glibc before 2.34 keeps in libdl.
        println!("cargo:rustc-link-lib=dylib=dl");
        // The vendored substrate is C++ (std::string, std::vector, the
        // vtables and typeinfo of its exception types). On MSVC the Rust link
        // already pulls the CRT that carries all of it; the GNU toolchain
        // links libstdc++ only when asked.
        println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
