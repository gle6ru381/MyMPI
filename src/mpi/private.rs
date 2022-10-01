pub(crate) use crate::{
    context::Context, debug, file_pos, p_mpi_check_type, p_mpi_type_size, reqqueue::RequestQueue,
    types::*, CHECK_RET,
};
pub use std::ffi::c_void;
pub use std::mem::MaybeUninit;
pub use std::ptr::null;
pub use std::ptr::null_mut;

pub const fn uninit<T>() -> T {
    unsafe { MaybeUninit::uninit().assume_init() }
}
