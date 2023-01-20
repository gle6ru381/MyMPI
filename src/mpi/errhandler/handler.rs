use std::ffi::CStr;

use crate::context::Context;
use crate::MPI_CHECK;
use crate::{shared::*, types::*, cstr};
use zstr::zstr;

type ErrHandler = fn(MPI_Comm, crate::types::MpiError);
const ERRH_MAX: usize = 2;

pub struct HandlerContext {
    handlers: [ErrHandler; ERRH_MAX],
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_COMM {
    ($comm:expr) => {
        Context::comm().check($comm)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_COMM {
    ($comm:expr) => {};
}

#[macro_export]
macro_rules! MPI_CHECK_COMM_RET {
    ($comm:expr) => {
        if cfg!(debug_assertions) {
            Err(Context::comm().check($comm))
        } else {
            Ok(())
        }
    };
}

impl HandlerContext {
    const ERR_STRINGS: &'static [*const i8] = &[
        cstr!("success"),
        cstr!("wrong buffer"),
        cstr!("wrong count"),
        cstr!("wrong type"),
        cstr!("wrong tah"),
        cstr!("wrong communicator"),
        cstr!("wrong rank"),
        cstr!("wrong request"),
        cstr!("wrong root"),
        cstr!("wrong reduction operation"),
        cstr!("wrong argument"),
        cstr!("unknown error"),
        cstr!("buffer truncated"),
        cstr!("other error"),
        cstr!("internal error"),
    ];

    pub const fn new() -> Self {
        HandlerContext {
            handlers: [super::callback::error_fatal, super::callback::error_return],
        }
    }

    pub fn err_to_string(err: i32) -> *const i8 {
        Self::ERR_STRINGS[err as usize]
    }

    pub fn check(comm: MPI_Comm, errh: MPI_Errhandler) -> MpiResult {
        crate::MPI_CHECK!(errh >= 0 && errh < ERRH_MAX as i32, comm, MPI_ERR_ARG)
    }

    pub fn call(&self, comm: MPI_Comm, code: crate::types::MpiError) -> crate::types::MpiError {
        let errh = Context::comm().err_handler(comm);
        debug_assert!(errh >= 0 && errh < ERRH_MAX as i32);

        self.handlers[errh as usize](comm, code);
        code
    }
}

#[no_mangle]
pub extern "C" fn MPI_Comm_call_errhandler(comm: MPI_Comm, code: crate::types::MpiError) -> i32 {
    Context::err_handler().call(comm, code);
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_class(code: i32, pclass: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(
        code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE as i32,
        MPI_COMM_WORLD,
        MPI_ERR_ARG
    );
    MPI_CHECK!(!pclass.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    unsafe { pclass.write(code) };

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_string(code: i32, str: *mut i8, plen: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(
        code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE as i32,
        MPI_COMM_WORLD,
        MPI_ERR_ARG
    );
    MPI_CHECK!(!str.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!plen.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    let len = unsafe {
        CStr::from_ptr(HandlerContext::err_to_string(code))
            .to_bytes_with_nul()
            .len()
    };
    unsafe {
        str.copy_from(HandlerContext::err_to_string(code), len);
        plen.write(len as i32);
    };

    MPI_SUCCESS
}
