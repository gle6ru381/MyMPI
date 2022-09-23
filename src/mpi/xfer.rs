use crate::types::*;
use crate::shm::MpiContext as Context;
use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn MPI_Send(buf : *const c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm) -> i32
{
    if !Context::is_init() || buf.is_null() || comm != MPI_COMM_WORLD {return !MPI_SUCCESS};

    return Context::send(buf, cnt, dtype, dest, tag, comm,);
}

#[no_mangle]
pub extern "C" fn MPI_Recv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, pstat : *mut MPI_Status) -> i32
{
    if !Context::is_init() || buf.is_null() || comm != MPI_COMM_WORLD {return !MPI_SUCCESS};

    return Context::recv(buf, cnt, dtype, src, tag, comm, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Isend(buf : *const c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Irecv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Test(preq : *mut MPI_Request, pflag : *mut i32, pstat : *mut MPI_Status) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Wait(preq : *mut MPI_Request, pstat : *mut MPI_Status) -> i32 {
    MPI_SUCCESS
}