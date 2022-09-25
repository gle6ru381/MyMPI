pub use std::ffi::c_void;

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

#[derive(Clone, Copy)]
pub struct MPI_Status {
    pub MPI_SOURCE : i32,
    pub MPI_TAG : i32,
    pub MPI_ERROR : i32,
    pub cnt : i32
}

impl Default for MPI_Status {
    fn default() -> Self {
        Self::new()
    }
}

impl MPI_Status {
    pub const fn new() -> Self {
        MPI_Status { MPI_SOURCE: 0, MPI_TAG: 0, MPI_ERROR: 0, cnt: 0 }
    }
}

#[derive(Clone, Copy)]
pub struct P_MPI_Request {
    pub buf : *mut c_void,
    pub stat : MPI_Status,
    pub comm : MPI_Comm,
    pub flag : i32,
    pub tag : i32,
    pub cnt : i32,
    pub rank : i32
}

impl Default for P_MPI_Request {
    fn default() -> Self {
        Self::new()
    }
}

impl P_MPI_Request {
    pub const fn new() -> Self {
        P_MPI_Request { buf: std::ptr::null_mut(), stat: MPI_Status::new(), comm: 0, flag: 0, tag: 0, cnt: 0, rank: 0 }
    }
}

pub type MPI_Request = *mut P_MPI_Request;

pub const MPI_UNDEFINED : i32 = -1;

pub const MPI_COMM_NULL : i32 = MPI_UNDEFINED;
pub const MPI_COMM_SELF : i32 = 0;
pub const MPI_COMM_WORLD : i32 = 1;

pub const MPI_BYTE : i32 = 1;
pub const MPI_INT : i32 = 4;
pub const MPI_DOUBLE : i32 = 8;

pub const MPI_MAX : i32 = 0;
pub const MPI_MIN : i32 = 1;
pub const MPI_SUM : i32 = 2;

pub const MPI_ERRORS_ARE_FATAL : i32 = 0;
pub const MPI_ERRORS_RETURN : i32 = 1;

pub const MPI_MAX_ERROR_STRING : i32 = 32;

pub const MPI_SUCCESS : i32 = 0;
pub const MPI_ERR_BUFFER : i32 = 1;
pub const MPI_ERR_COUNT : i32 = 2;
pub const MPI_ERR_TYPE : i32 = 3;
pub const MPI_ERR_TAG : i32 = 4;
pub const MPI_ERR_COMM : i32 = 5;
pub const MPI_ERR_RANK : i32 = 6;
pub const MPI_ERR_REQUEST : i32 = 7;
pub const MPI_ERR_ROOT : i32 = 8;
pub const MPI_ERR_OP : i32 = 9;
pub const MPI_ERR_ARG : i32 = 10;
pub const MPI_ERR_UNKNOWN : i32 = 11;
pub const MPI_ERR_TRUNCATE : i32 = 12;
pub const MPI_ERR_OTHER : i32 = 13;
pub const MPI_ERR_INTERN : i32 = 14;
pub const MPI_ERR_LASTCODE : i32 = 14;