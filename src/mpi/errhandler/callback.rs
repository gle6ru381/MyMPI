use libc::exit;

use crate::shared::*;
use crate::debug_core;

type Callback = fn(MPI_Comm, crate::types::MpiError);

pub fn error_fatal(pcomm: MPI_Comm, pcode: crate::types::MpiError) {
    debug_core!("Error", "Fatal error, code: {}", pcode as i32);
    unsafe {exit(-1)};
}

pub fn error_return(_: MPI_Comm, pcode: crate::types::MpiError) {
    todo!();
}