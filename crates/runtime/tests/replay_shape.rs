//! The DFlash2 replay examples' load shape and its command-line flags (GitHub
//! #340), on CPU: the examples need `cuda`, their shape does not. The tests
//! are the module's own (`examples/support/replay_shape.rs`).

#[path = "../examples/support/replay_shape.rs"]
mod replay_shape;

#[path = "../examples/support/replay_identity.rs"]
mod replay_identity;
