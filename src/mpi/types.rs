use crate::xfer::request::Request;
pub use std::ffi::c_void;
use std::mem::MaybeUninit;

pub type MPI_Datatype = i32;
pub type MPI_Comm = i32;
pub type MPI_Op = i32;
pub type MPI_Errhandler = i32;

#[macro_export]
macro_rules! cstr {
    ($cstr:literal) => {
        zstr!($cstr).as_ptr()
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MPI_Status {
    pub MPI_SOURCE: i32,
    pub MPI_TAG: i32,
    pub MPI_ERROR: i32,
    pub cnt: i32,
}

impl Default for MPI_Status {
    fn default() -> Self {
        Self::new()
    }
}

impl MPI_Status {
    pub const fn new() -> Self {
        MPI_Status {
            MPI_SOURCE: 0,
            MPI_TAG: 0,
            MPI_ERROR: 0,
            cnt: 0,
        }
    }

    pub const fn uninit() -> Self {
        #![allow(invalid_value)]
        unsafe { MaybeUninit::uninit().assume_init() }
    }
}

pub type MPI_Request = *mut Request;

pub const MPI_UNDEFINED: i32 = -1;

pub const MPI_COMM_NULL: i32 = MPI_UNDEFINED;
pub const MPI_COMM_SELF: i32 = 0;
pub const MPI_COMM_WORLD: i32 = 1;

pub const MPI_BYTE: i32 = 1;
pub const MPI_INT: i32 = 4;
pub const MPI_DOUBLE: i32 = 8;

pub const MPI_MAX: i32 = 0;
pub const MPI_MIN: i32 = 1;
pub const MPI_SUM: i32 = 2;

pub const MPI_ERRORS_ARE_FATAL: i32 = 0;
pub const MPI_ERRORS_RETURN: i32 = 1;

pub const MPI_MAX_ERROR_STRING: i32 = 32;

#[derive(Debug, Clone, Copy)]
pub enum MpiError {
    MPI_ERR_BUFFER = 1,
    MPI_ERR_COUNT,
    MPI_ERR_TYPE,
    MPI_ERR_TAG,
    MPI_ERR_COMM,
    MPI_ERR_RANK,
    MPI_ERR_REQUEST,
    MPI_ERR_ROOT,
    MPI_ERR_OP,
    MPI_ERR_ARG,
    MPI_ERR_UNKNOWN,
    MPI_ERR_TRUNCATE,
    MPI_ERR_OTHER,
    MPI_ERR_INTERN,
    MPI_ERR_PENDING,
    MPI_ERR_IN_STATUS,
    MPI_ERR_LASTCODE,
}

pub type MpiResult = Result<(), MpiError>;

pub const MPI_SUCCESS: i32 = 0;
