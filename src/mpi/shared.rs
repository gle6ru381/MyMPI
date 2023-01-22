pub(crate) use crate::{
    backend::reqqueue::RequestQueue, context::Context, type_size, types::MpiError::*,
    types::MpiResult, types::*,
};

pub use std::ffi::c_void;
pub use std::mem::MaybeUninit;
pub use std::ptr::null;
pub use std::ptr::null_mut;

pub const fn uninit<T>() -> T {
    unsafe { MaybeUninit::uninit().assume_init() }
}
