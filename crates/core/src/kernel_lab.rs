//! Offline kernel experiments. The server has no route to this module.
use serde_json::Value;

/// Bound an input before it reaches an FFI function with caller-sized output.
pub fn tokens(value: &Value, max_len: usize, vocab: u32) -> Result<Vec<i32>, String> {
    let values = value.as_array().ok_or("tokens must be an array")?;
    if values.is_empty() || values.len() > max_len { return Err("token count outside context".into()); }
    values.iter().map(|v| {
        v.as_u64().filter(|&id| id < u64::from(vocab)).map(|id| id as i32)
            .ok_or_else(|| "token id outside vocabulary".into())
    }).collect()
}

/// Replace model-local code and recapture graphs without moving weights/pool.
///
/// # Safety
/// The library is trusted native code implementing the candidate ABI. The
/// caller must own all work on this CUDA device and use this model's pool.
#[cfg(feature = "cuda")]
pub unsafe fn reload(
    model: &mut crate::model_load::Model, pool: &mut crate::seq::SeqPool, path: Option<&std::path::Path>,
) -> Result<(), String> {
    use std::ffi::{CStr, CString};
    unsafe extern "C" {
        fn ignis_model_reload_trellis(model: *mut crate::model_load::ffi::IgnisModel,
            pool: *mut crate::seq::ffi::IgnisSeqPool, path: *const std::ffi::c_char) -> i32;
        fn ignis_trellis_reload_error() -> *const std::ffi::c_char;
    }
    let path = path.map(|p| CString::new(p.as_os_str().as_encoded_bytes()))
        .transpose().map_err(|_| "candidate path contains NUL")?;
    let rc = unsafe { ignis_model_reload_trellis(model.handle(), pool.handle(),
        path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr())) };
    if rc == 0 { Ok(()) } else {
        Err(unsafe { CStr::from_ptr(ignis_trellis_reload_error()) }.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn malformed_replay_never_reaches_the_leaf() {
        for v in [json!([]), json!([-1]), json!([20]), json!([1.5]), json!(["1"]), json!([0, 1, 2])] {
            assert!(tokens(&v, 2, 20).is_err(), "{v}");
        }
        assert_eq!(tokens(&json!([0, 19]), 2, 20).unwrap(), [0, 19]);
    }
}
