pub(crate) use crate::{
    backend::reqqueue::RequestQueue, context::Context, debug_print, file_pos, p_mpi_check_type,
    p_mpi_type_size, types::*, CHECK_RET, MPI_CHECK_COMM_RET, MPI_CHECK_RET,
};
pub use std::ffi::c_void;
pub use std::mem::MaybeUninit;
pub use std::ptr::null;
pub use std::ptr::null_mut;

pub const fn uninit<T>() -> T {
    unsafe { MaybeUninit::uninit().assume_init() }
}
