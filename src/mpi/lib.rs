#![allow(non_camel_case_types, non_snake_case)]

pub mod types;
mod shm;

use shm::MpiContext as Context;

pub use types::*;
pub use std::ffi::c_void;

#[no_mangle]
pub extern "C" fn MPI_Init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32
{
    println!("Enter mpi init");
    if Context::is_init() || pargc.is_null() || pargv.is_null() {
            println!("MPI init fail {}, {}, {}", Context::is_init(), pargc.is_null(), pargv.is_null());
            return !MPI_SUCCESS;
    }

    if Context::parseArgs(pargc, pargv) != MPI_SUCCESS || Context::allocate() != MPI_SUCCESS || Context::mpi_split(0, Context::size()) != MPI_SUCCESS {
        println!("{}/{}: -> failure", Context::rank(), Context::size());
        return !MPI_SUCCESS;
    }

    Context::init();

    println!("{}/{}: -> success", Context::rank(), Context::size());

    return MPI_SUCCESS;
}

#[no_mangle]
pub extern "C" fn MPI_Finalize() -> i32
{
    if Context::is_init() {return MPI_SUCCESS};

    Context::deinit();

    println!("{}/{}: -> finalize", Context::rank(), Context::size());

    return MPI_SUCCESS;
}

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
pub extern "C" fn MPI_Send(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm) -> i32
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
pub extern "C" fn MPI_Wtime() -> f64
{
    let mut tv = std::mem::MaybeUninit::<libc::timeval>::uninit();
    unsafe {
        libc::gettimeofday(tv.as_mut_ptr(), std::ptr::null_mut());
        let res = tv.assume_init();
        return res.tv_sec as f64 + 0.000001 * res.tv_usec as f64;
    }
}