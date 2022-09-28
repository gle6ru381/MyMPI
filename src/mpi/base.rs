use crate::{types::*, private::*, MPI_CHECK, MPI_CHECK_COMM};
use crate::context::Context;
use crate::errhandle::*;
use crate::comm::*;

pub (crate) fn p_mpi_abort(comm : MPI_Comm, code : i32) {
    Context::deinit();

    std::process::exit(-1);
}

#[no_mangle]
pub extern "C" fn MPI_Init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32
{
    println!("Enter mpi init");
    if Context::is_init() || pargc.is_null() || pargv.is_null() {
            println!("MPI init fail {}, {}, {}", Context::is_init(), pargc.is_null(), pargv.is_null());
            return !MPI_SUCCESS;
    }

    MPI_CHECK!(!Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(!pargc.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pargv.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    println!("{}/{}: -> success", Context::rank(), Context::size());

    return Context::init(pargc, pargv);
}

#[no_mangle]
pub extern "C" fn MPI_Finalize() -> i32
{
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);

    let mut code = p_mpi_comm_finit();
    if code != MPI_SUCCESS {
        return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
    }

    code = Context::deinit();
    if code != MPI_SUCCESS {
        return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
    }

    code = p_mpi_errh_fini();
    if code != MPI_SUCCESS {
        return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
    }

    println!("{}/{}: -> finalize", Context::rank(), Context::size());

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Abort(comm : MPI_Comm, code : i32) -> i32
{
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);

    p_mpi_abort(comm, code);

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