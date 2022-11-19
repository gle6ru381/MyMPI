use crate::context::Context;
use crate::{private::*, types::*, MPI_CHECK, MPI_CHECK_COMM};

pub(crate) fn p_mpi_abort(_: MPI_Comm, _: i32) {
    Context::deinit();

    panic!();
}

#[no_mangle]
pub extern "C" fn MPI_Init(pargc: *mut i32, pargv: *mut *mut *mut i8) -> i32 {
    debug!("Enter mpi init");

    MPI_CHECK!(!Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    //MPI_CHECK!(!pargc.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    //MPI_CHECK!(!pargv.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    debug!("-> success");

    return Context::init(pargc, pargv);
}

#[no_mangle]
pub extern "C" fn MPI_Finalize() -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    debug!("MPI finalize");

    let code;

    code = Context::deinit();
    if code != MPI_SUCCESS {
        Context::call_error(MPI_COMM_WORLD, code);
    }

    debug_1!("Finalize");

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Abort(comm: MPI_Comm, code: i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);

    p_mpi_abort(comm, code);

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Wtime() -> f64 {
    let mut tv = std::mem::MaybeUninit::<libc::timeval>::uninit();
    unsafe {
        libc::gettimeofday(tv.as_mut_ptr(), std::ptr::null_mut());
        let res = tv.assume_init();
        return res.tv_sec as f64 + 0.000001 * res.tv_usec as f64;
    }
}
