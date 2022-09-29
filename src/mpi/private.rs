pub(crate) use crate::{
    context::Context, debug, file_pos, p_mpi_call_errhandler, p_mpi_check_comm, p_mpi_check_errh,
    p_mpi_check_rank, p_mpi_check_type, p_mpi_dec_key, p_mpi_inc_key, p_mpi_rank_map,
    p_mpi_rank_unmap, p_mpi_tag_map, p_mpi_tag_unmap, p_mpi_type_size, reqqueue::RequestQueue,
    types::*, CHECK_RET,
};
pub use std::ffi::c_void;
pub use std::mem::MaybeUninit;
pub use std::ptr::null;
pub use std::ptr::null_mut;

#[macro_export]
macro_rules! uninit {
    () => {
        unsafe { MaybeUninit::uninit().assume_init() }
    };
}
