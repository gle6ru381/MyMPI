use crate::types::*;

#[no_mangle]
pub extern "C" fn MPI_Comm_call_errhandler(comm : MPI_Comm, code : i32) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_class(code : i32, pclass : *mut i32) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Error_string(code : i32, str : *mut i8, plen : *mut i32) -> i32 {
    MPI_SUCCESS
}