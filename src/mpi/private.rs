pub (crate) use crate::{p_mpi_call_errhandler, p_mpi_check_comm, p_mpi_check_errh};

#[macro_export]
macro_rules! uninit {
    () => {
       unsafe {MaybeUninit::uninit().assume_init()}
    };
}