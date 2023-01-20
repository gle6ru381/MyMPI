use crate::bcast::BCAST_IMPL;
use crate::shared::*;
use crate::{uninit, MPI_Comm, MPI_Datatype, MPI_Request};
use crate::{MPI_CHECK};
use libc::c_void;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use crate::xfer::request::Request;
use crate::xfer::ppp::recv::{irecv, recv};
use crate::xfer::ppp::send::{isend, send};

#[no_mangle]
pub extern "C" fn MPI_Isend(
    buf: *const c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    dest: i32,
    tag: i32,
    comm: MPI_Comm,
    preq: *mut MPI_Request,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    MPI_CHECK!(!preq.is_null(), comm,MPI_ERR_ARG);
    let mut req: &mut Request = uninit();
    unsafe {
        if let Err(code) = isend(
            from_raw_parts(buf as *const u8, dataLen as usize),
            dest,
            tag,
            comm,
            &mut req,
        ) {
            return code as i32;
        } else {
            *preq = req;
            return MPI_SUCCESS;
        }
    }
}

#[no_mangle]
pub extern "C" fn MPI_Irecv(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    src: i32,
    tag: i32,
    comm: MPI_Comm,
    preq: *mut MPI_Request,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    MPI_CHECK!(!preq.is_null(), comm, MPI_ERR_ARG);
    unsafe {
        if let Err(code) = irecv(
            from_raw_parts_mut(buf as *mut u8, dataLen as usize),
            src,
            tag,
            comm,
            &mut &mut**preq,
        ) {
            return code as i32;
        } else {
            return MPI_SUCCESS;
        }
    }
}

#[no_mangle]
pub extern "C" fn MPI_Send(
    buf: *const c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    dest: i32,
    tag: i32,
    comm: MPI_Comm,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    MPI_CHECK!(!buf.is_null(), comm,MPI_ERR_ARG);
    let result;
    unsafe {
        result = send(from_raw_parts(buf as *const u8, dataLen as usize), dest, tag, comm);
    }
    if let Err(code) = result {
        return code as i32;
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Recv(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    src: i32,
    tag: i32,
    comm: MPI_Comm,
    pstat: *mut MPI_Status,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    let result = unsafe {recv(from_raw_parts_mut(buf as *mut u8, dataLen as usize), src, tag, comm, pstat.as_mut())};
    if let Err(code) = result {
        return code as i32;
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Sendrecv(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    dest: i32,
    stag: i32,
    rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    src: i32,
    rtag: i32,
    comm: MPI_Comm,
    pstat: *mut MPI_Status,
) -> i32 {
    MPI_CHECK!(Context::is_init(), comm, MPI_ERR_OTHER);
    MPI_CHECK!(!pstat.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    let mut req: [MPI_Request; 2] = uninit();
    let mut stat: [MPI_Status; 2] = uninit();

    todo!();

    // CHECK_RET!(MPI_Isend(sbuf, scnt, sdtype, dest, stag, comm, &mut req[0]));
    // CHECK_RET!(MPI_Irecv(rbuf, rcnt, rdtype, src, rtag, comm, &mut req[1]));
    // CHECK_RET!(MPI_Waitall(2, req.as_mut_ptr(), stat.as_mut_ptr()));

    unsafe {
        pstat.write(stat[1]);
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Test(preq: *mut MPI_Request, pflag: *mut i32, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pflag.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    if let Err(code) = unsafe {Request::test(*preq, &mut *pflag, pstat.as_mut())} {
        return code as i32;
    } else {
        return MPI_SUCCESS;
    }
}

#[no_mangle]
pub extern "C" fn MPI_Wait(preq: *mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    if let Err(code) = unsafe { (**preq).wait(pstat.as_mut()) } {
        return code as i32;
    } else {
        return MPI_SUCCESS;
    }
}

#[no_mangle]
pub extern "C" fn MPI_Waitall(cnt: i32, preq: *mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(
        !preq.is_null() && !pstat.is_null(),
        MPI_COMM_WORLD,
        MPI_ERR_ARG
    );
    MPI_CHECK!(cnt >= 0, MPI_COMM_WORLD, MPI_ERR_COUNT);
    if cnt == 0 {
        return MPI_SUCCESS;
    }
    if let Err(code) = Request::wait_all(
        unsafe { std::slice::from_raw_parts_mut(preq, cnt as usize) },
        unsafe { std::slice::from_raw_parts_mut(pstat, cnt as usize) },
    ) {
        return code as i32;
    } else {
        return MPI_SUCCESS;
    }
}

#[no_mangle]
pub extern "C" fn MPI_Barrier(comm: MPI_Comm) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Bcast(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    if let Err(code) = BCAST_IMPL(
        unsafe { from_raw_parts_mut(buf as *mut u8, dataLen as usize) },
        root,
        comm,
    ) {
        return code as i32;
    }
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Reduce(
    sbuf: *const c_void,
    mut rbuf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    op: MPI_Op,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    todo!()
}

#[no_mangle]
pub extern "C" fn MPI_Allreduce(
    sbuf: *const c_void,
    rbuf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    op: MPI_Op,
    comm: MPI_Comm,
) -> i32 {
    todo!()
}

#[no_mangle]
pub extern "C" fn MPI_Gather(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    mut rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Allgather(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    comm: MPI_Comm,
) -> i32 {
    todo!()
}


#[no_mangle]
pub extern "C" fn MPI_Comm_size(comm: MPI_Comm, psize: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!psize.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        psize.write(Context::comm_size(comm));
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_rank(comm: MPI_Comm, prank: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!prank.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        prank.write(Context::comm_rank(comm));
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_dup(comm: MPI_Comm, pcomm: *mut MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = Context::comm().comm_dup(comm, pcomm);
    if code != MPI_SUCCESS {
        Context::err_handler().call(comm, MPI_ERR_OTHER);
    }

    code
}

#[no_mangle]
pub extern "C" fn MPI_Comm_split(comm: MPI_Comm, col: i32, key: i32, pcomm: *mut MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    MPI_CHECK!(col >= 0 || col == MPI_UNDEFINED, comm, MPI_ERR_ARG);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = Context::comm().comm_split(comm, col, key, pcomm);
    if let Err(code) = code {
        Context::err_handler().call(comm, code);
        return code as i32;
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_get_errhandler(comm: MPI_Comm, perrh: *mut MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!perrh.is_null(), comm, MPI_ERR_ARG);

    unsafe { perrh.write(Context::comm().err_handler(comm)) }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_set_errhandler(comm: MPI_Comm, errh: MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    crate::MPI_CHECK_COMM!(comm);
    //  MPI_CHECK_ERRH!(comm, errh);

    Context::comm().set_err_handler(comm, errh);

    MPI_SUCCESS
}
