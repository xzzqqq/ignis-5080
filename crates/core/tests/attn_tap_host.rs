//! Run the tap's host scoring references without CUDA. The capture FFI is
//! declared by the included module but no test arms or calls the device tap.
#![allow(dead_code)]

pub use ignis_core::KvFormat;

#[path = "../src/attn_tap.rs"]
mod attn_tap;
