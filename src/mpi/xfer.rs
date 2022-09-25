use crate::{types::*, uninit};
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