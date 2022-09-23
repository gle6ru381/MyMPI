use crate::types::*;
use crate::shm::MpiContext as Context;

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
pub extern "C" fn MPI_Abort(comm : MPI_Comm, code : i32) -> i32
{
    MPI_SUCCESS
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