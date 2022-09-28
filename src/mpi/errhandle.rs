use std::ffi::CStr;

use crate::context::Context;
use crate::{types::*, private::*, cstr, p_mpi_abort, CommGroup};
use zstr::zstr;

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK {
    ($exp:expr, $comm:expr, $code:expr) => {
        if $exp {
            MPI_SUCCESS
        } else {
            debug!("Chech failed");
            p_mpi_call_errhandler($comm, $code)
        }
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK {
    ($expr:expr, $comm:expr, $code:expr) => {
        MPI_SUCCESS
    }
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_COMM {
    ($comm:expr) => {
        p_mpi_check_comm($comm)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_COMM {
    ($comm:expr) => {
        MPI_SUCCESS
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
    () => {
        
    };
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_RANK {
    ($rank:expr, $comm:expr) => {
        p_mpi_check_rank($rank, $comm)
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
        p_mpi_check_type($dtype, $comm)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_TYPE {
    ($dtype:expr, $comm:expr) => {
        MPI_SUCCESS
    };
}

#[macro_export]
#[cfg(debug_assertions)]
macro_rules! MPI_CHECK_OP {
    ($op:expr, $comm:expr) => {
        p_mpi_check_op($op, $comm)
    };
}

#[macro_export]
#[cfg(not(debug_assertions))]
macro_rules! MPI_CHECK_OP {
    ($op:expr, $comm:expr) => {
        MPI_SUCCESS
    };
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

const ERR_STRINGS : &'static [*const i8] = &[
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
    cstr!("internal error")];

type ErrHandler = fn(&MPI_Comm, &mut i32);
const ERRH_MAX : i32 = 2;

pub (crate) fn p_mpi_errors_are_fatal(pcomm : &MPI_Comm, pcode : &mut i32) {
    println!("MPI fatal error for {}, code: {pcode}", Context::rank());
    p_mpi_abort(*pcomm, *pcode);
}

pub (crate) fn p_mpi_errors_return(pcomm : &MPI_Comm, pcode : &mut i32) {

}

static ERRH : [ErrHandler; ERRH_MAX as usize] = [p_mpi_errors_are_fatal, p_mpi_errors_return];

pub (crate) fn p_mpi_check_errh(comm : MPI_Comm, errh : MPI_Errhandler) {
    MPI_CHECK!(errh >= 0 && errh < ERRH_MAX, comm, MPI_ERR_ARG);
}

pub (crate) fn p_mpi_call_errhandler(comm : MPI_Comm, code : i32) -> i32 {
    let errh = CommGroup::err_handler(comm);

    debug_assert!(errh >= 0 && errh < ERRH_MAX);

    let mut ret = code;

    ERRH[errh as usize](&comm, &mut ret);

    ret
}

pub (crate) fn p_mpi_errh_init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
    debug_assert!(!Context::is_init());
    debug_assert!(!pargc.is_null());
    debug_assert!(!pargv.is_null());

    MPI_SUCCESS
}

pub (crate) fn p_mpi_errh_fini() -> i32 {
    debug_assert!(Context::is_init());

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_call_errhandler(comm : MPI_Comm, code : i32) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_class(code : i32, pclass : *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE, MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pclass.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    unsafe {pclass.write(code)};

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_string(code : i32, str : *mut i8, plen : *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(code >= MPI_SUCCESS && code <= MPI_ERR_LASTCODE, MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!str.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!plen.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    let len = unsafe {CStr::from_ptr(ERR_STRINGS[code as usize]).to_bytes_with_nul().len() };
    unsafe {
        str.copy_from(ERR_STRINGS[code as usize], len);
        plen.write(len as i32);
    };

    MPI_SUCCESS
}