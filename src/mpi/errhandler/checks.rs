use crate::{MPI_Comm, MpiError};

#[macro_export]
macro_rules! MPI_CHECK {
    ($exp:expr, $comm:expr, $code:expr) => {
        if cfg!(debug_assertions) && !$exp {
            crate::debug_core!("Check", "Check failed");
            Err(Context::err_handler().call($comm, $code))
        } else {
            Ok(())
        }
    };
}

#[allow(dead_code)]
pub fn check_handler(comm: MPI_Comm, handler: MpiError) {
    todo!()
}
