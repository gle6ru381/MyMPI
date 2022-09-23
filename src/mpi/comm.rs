use crate::types::*;
use crate::shm::MpiContext as Context;

#[no_mangle]
pub extern "C" fn MPI_Comm_size(comm : MPI_Comm, psize : *mut i32) -> i32
{
    if !Context::is_init() || comm != MPI_COMM_WORLD || psize.is_null() {return !MPI_SUCCESS};

    unsafe {
        psize.write(Context::size());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_rank(comm : MPI_Comm, prank : *mut i32) -> i32
{
    if !Context::is_init() || comm != MPI_COMM_WORLD || prank.is_null() {return !MPI_SUCCESS};

    unsafe {
        prank.write(Context::rank());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_dup(comm : MPI_Comm, pcomm : *mut MPI_Comm) -> i32
{
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_split(comm : MPI_Comm, col : i32, key : i32, pcomm : *mut MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_get_errhandler(comm : MPI_Comm, perrh : *mut MPI_Errhandler) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_set_errhandler(comm : MPI_Comm, errh : MPI_Errhandler) -> i32 {
    MPI_SUCCESS
}