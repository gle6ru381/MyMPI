pub (crate) use crate::{types::*, context::Context, reqqueue::RequestQueue, p_mpi_call_errhandler, p_mpi_check_comm, p_mpi_check_errh, p_mpi_check_rank, p_mpi_check_type, p_mpi_type_size};
pub use std::ffi::c_void;
pub use std::ptr::null;
pub use std::ptr::null_mut;

#[macro_export]
macro_rules! uninit {
    () => {
       unsafe {MaybeUninit::uninit().assume_init()}
    };
}