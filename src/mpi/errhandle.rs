use std::ffi::CStr;

use crate::context::Context;
use crate::{cstr, p_mpi_abort, private::*, types::*};
use zstr::zstr;

type ErrHandler = fn(MPI_Comm, i32);
const ERRH_MAX: usize = 2;

pub struct HandlerContext {
    handlers: [ErrHandler; ERRH_MAX],
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK {
    ($exp:expr, $comm:expr, $code:expr) => {
        if !$exp {
            debug!("Check failed");
            Context::err_handler().call($comm, $code);
        }
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK {
    ($expr:expr, $comm:expr, $code:expr) => {};
}

#[macro_export]
macro_rules! MPI_CHECK_RET {
    ($exp:expr, $comm:expr, $code:expr) => {
        if cfg!(debug_assertions) {
            if $exp {
                MPI_SUCCESS
            } else {
                debug!("Check failed");
                Context::err_handler().call($comm, $code)
            }
        } else {
            MPI_SUCCESS
        }
    };
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_COMM {
    ($comm:expr) => {
        Context::comm().check($comm);
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
            Context::comm().check($comm)
        } else {
            MPI_SUCCESS
        }
    };
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_ERRH {
    ($comm:tt, $errh:tt) => {
        p_mpi_check_errh($comm, $errh)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_ERRH {
    () => {};
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_RANK {
    ($rank:expr, $comm:expr) => {
        Context::comm().check_rank($rank, $comm)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_RANK {
    ($rank:expr, $comm:expr) => {
        MPI_SUCCESS
    };
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_TYPE {
    ($dtype:expr, $comm:expr) => {
        p_mpi_check_type($dtype, $comm);
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_TYPE {
    ($dtype:expr, $comm:expr) => {};
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_OP {
    ($op:expr, $comm:expr) => {
        p_mpi_check_op($op, $comm);
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_OP {
    ($op:expr, $comm:expr) => {};
}

#[macro_export]
macro_rules! CHECK_RET {
    ($e:expr) => {
        let code = $e;
        if code != MPI_SUCCESS {
            return code;
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
            handlers: [p_mpi_errors_are_fatal, p_mpi_errors_return],
        }
    }

    pub fn err_to_string(err: i32) -> *const i8 {
        Self::ERR_STRINGS[err as usize]
    }

    pub fn check(comm: MPI_Comm, errh: MPI_Errhandler) -> i32 {
        MPI_CHECK_RET!(errh >= 0 && errh < ERRH_MAX as i32, comm, MPI_ERR_ARG)
    }

    pub fn call(&self, comm: MPI_Comm, code: MPI_Errhandler) -> i32 {
        let errh = Context::comm().err_handler(comm);
        debug_assert!(errh >= 0 && errh < ERRH_MAX as i32);

        self.handlers[errh as usize](comm, code);
        code
    }
}

fn p_mpi_errors_are_fatal(pcomm: MPI_Comm, pcode: i32) {
    println!("MPI fatal error for {}, code: {pcode}", Context::rank());
    p_mpi_abort(pcomm, pcode);
}

fn p_mpi_errors_return(_: MPI_Comm, pcode: i32) {
    println!("MPI simple error for {}, code: {pcode}", Context::rank())
}

#[no_mangle]
pub extern "C" fn MPI_Comm_call_errhandler(comm: MPI_Comm, code: i32) -> i32 {
    Context::err_handler().call(comm, code);
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_class(code: i32, pclass: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(
        code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE,
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
        code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE,
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
