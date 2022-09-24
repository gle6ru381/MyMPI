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

pub struct MPI_Status {
    pub MPI_SOURCE : i32,
    pub MPI_TAG : i32,
    pub MPI_ERROR : i32,
    pub cnt : i32
}

pub struct MPI_Request {
    buf : *mut c_void,
    stat : MPI_Status,
    comm : MPI_Comm,
    flag : i32,
    tag : i32,
    cnt : i32,
    rank : i32
}

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