use crate::{types::*, private::*, MPI_CHECK};
use std::ffi::c_void;

pub (crate) fn p_mpi_check_op(op : MPI_Op, comm : MPI_Comm) -> i32 {
    MPI_CHECK!(op == MPI_MAX || op == MPI_MIN || op == MPI_SUM, comm, MPI_ERR_OP)
}

#[no_mangle]
pub extern "C" fn MPI_Barrier(comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Bcast(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, root : i32, comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Reduce(sbuf : *const c_void, rbuf : *mut c_void, cnt : i32, dtype : MPI_Datatype, op : MPI_Op, root : i32, comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Allreduce(sbuf : *const c_void, rbuf : *mut c_void, cnt : i32, dtype : MPI_Datatype, op : MPI_Op, comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Gather(sbuf : *const c_void, scnt : i32, sdtype : MPI_Datatype, rbuf : *mut c_void, rcnt : i32, rdtype : MPI_Datatype, root : i32, comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Allgather(sbuf : *const c_void, scnt : i32, sdtype : MPI_Datatype, rbuf : *mut c_void, rcnt : i32, rdtype : MPI_Datatype, comm : MPI_Comm) -> i32 {
    MPI_SUCCESS
}