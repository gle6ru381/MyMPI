use crate::{types::*, private::*, uninit, MPI_CHECK};
use crate::context::Context;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::{null_mut, null};

#[no_mangle]
pub extern "C" fn MPI_Send(buf : *const c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm) -> i32
{
    let mut req : MPI_Request = uninit!();
    let mut code = MPI_Isend(buf, cnt, dtype, dest, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    println!("Send isend finish");
    code = MPI_Wait(&mut req, null_mut());
    if code != MPI_SUCCESS {
        return code;
    }

    println!("Send finish");

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Recv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, pstat : *mut MPI_Status) -> i32
{
    let mut req : MPI_Request = uninit!();
    let mut code = MPI_Irecv(buf, cnt, dtype, src, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    println!("Recv irecv finish");
    code = MPI_Wait(&mut req, pstat);
    if code != MPI_SUCCESS {
        return code;
    }

    println!("Recv finish");

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Sendrecv(sbuf : *const c_void, scnt : i32, sdtype : MPI_Datatype, dest : i32, stag : i32, rbuf : *mut c_void, rcnt : i32, rdtype : MPI_Datatype, src : i32, rtag : i32, comm : MPI_Comm, pstat : *mut MPI_Status) -> i32 
{
    MPI_CHECK!(Context::is_init(), comm, MPI_ERR_OTHER);
    MPI_CHECK!(!pstat.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    let mut req : [MPI_Request; 2] = uninit!();
    let mut stat : [MPI_Status; 2] = uninit!();

    CHECK_RET!(MPI_Isend(sbuf, scnt, sdtype, dest, stag, comm, &mut req[0]));
    CHECK_RET!(MPI_Irecv(rbuf, rcnt, rdtype, src, rtag, comm, &mut req[1]));
    CHECK_RET!(MPI_Waitall(2, req.as_mut_ptr(), stat.as_mut_ptr()));

    unsafe {
        pstat.write(stat[1]);
    }
    
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Isend(buf : *const c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
    Context::send(buf, cnt, dtype, dest, tag, comm, preq)
}

#[no_mangle]
pub extern "C" fn MPI_Irecv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
    Context::recv(buf, cnt, dtype, src, tag, comm, preq)
}

#[no_mangle]
pub extern "C" fn MPI_Test(preq : *mut MPI_Request, pflag : *mut i32, pstat : *mut MPI_Status) -> i32 {
    Context::test(preq, pflag, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Wait(preq : *mut MPI_Request, pstat : *mut MPI_Status) -> i32 {
    Context::wait(preq, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Waitall(cnt : i32, preq : *mut MPI_Request, pstat : *mut MPI_Status) -> i32 {
    Context::wait_all(cnt, preq, pstat)
}