use libc::exit;

use crate::debug_core;
use crate::shared::*;

type Callback = fn(MPI_Comm, crate::types::MpiError);

pub fn error_fatal(_: MPI_Comm, pcode: crate::types::MpiError) {
    debug_core!("Error", "Fatal error, code: {}", pcode as i32);
    unsafe { exit(-1) };
}

pub fn error_return(_: MPI_Comm, _: crate::types::MpiError) {
    todo!();
}
