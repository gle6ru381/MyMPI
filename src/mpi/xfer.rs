use crate::context::Context;
use crate::object::types::Typed;
use crate::{shared::*, MPI_CHECK};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::slice::{from_raw_parts, from_raw_parts_mut};

impl P_MPI_Request {
    pub fn test(preq: &mut MPI_Request, pflag: &mut i32, pstat: *mut MPI_Status) -> i32 {
        let code = Context::progress();
        if code != MPI_SUCCESS {
            return Context::err_handler().call(MPI_COMM_WORLD, code);
        }
        *pflag = 0;
        if !(*preq).is_null() {
            let req = unsafe { &mut **preq };
            if req.flag != 0 {
                debug!("Find request with tag: {}, rank: {}", req.tag, req.rank);
                *pflag = 1;

                if !pstat.is_null() {
                    let stat = unsafe { &mut *pstat };
                    stat.MPI_SOURCE = Context::comm().rank_unmap(req.comm, req.stat.MPI_SOURCE);
                    stat.MPI_TAG = Context::comm().tag_unmap(req.comm, req.stat.MPI_TAG);
                    stat.cnt = req.stat.cnt;
                }
                req.flag = 0;
                Context::shm().free_req(req, *preq);
            }
        }
        MPI_SUCCESS
    }

    pub fn wait(preq: &mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
        let mut flag = 0;
        while flag == 0 {
            let code = Self::test(preq, &mut flag, pstat);
            if code != MPI_SUCCESS {
                return code;
            }
        }
        MPI_SUCCESS
    }

    pub fn wait_all(reqs: &mut [MPI_Request], pstat: &mut [MPI_Status]) -> i32 {
        debug_assert!(reqs.len() == pstat.len());
        debug!("Wait all, size: {}", reqs.len());

        for i in pstat.iter_mut() {
            i.MPI_ERROR = MPI_ERR_PENDING;
        }

        let mut flag = 0;
        let mut flags = 0;

        while flags != reqs.len() as i32 {
            for (stat, req) in pstat.iter_mut().zip(reqs.iter_mut()) {
                if stat.MPI_ERROR == MPI_ERR_PENDING {
                    let code = Self::test(req, &mut flag, stat);
                    if code != MPI_SUCCESS {
                        stat.MPI_ERROR = code;
                        return MPI_ERR_IN_STATUS;
                    }
                    if flag != 0 {
                        stat.MPI_ERROR = MPI_SUCCESS;
                        flags += 1;
                    }
                }
            }
        }
        MPI_SUCCESS
    }
}

pub(crate) fn send<T: Typed>(
    buf: &[T],
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
    req: &mut MPI_Request,
) -> i32 {
    let dest = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(dest != Context::rank(), comm, MPI_ERR_INTERN);
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_TAG);

    let tag = Context::comm().tag_map(comm, tag);
    debug!("Send call to {dest} with tag {tag}");

    let code = Context::progress();
    if code != MPI_SUCCESS {
        return Context::err_handler().call(comm, code);
    }

    let new_req = Context::shm().get_send();
    if let Some(r) = new_req {
        *req = r;
        unsafe {
            **req = P_MPI_Request {
                buf: buf.as_ptr() as *mut T as *mut c_void,
                stat: MPI_Status::new(),
                comm,
                flag: 0,
                tag,
                cnt: buf.len() as i32 * p_mpi_type_size(T::into_mpi()),
                rank: dest,
            }
        };
    } else {
        *req = null_mut();
        Context::err_handler().call(comm, MPI_ERR_INTERN);
        return MPI_ERR_INTERN;
    }
    MPI_SUCCESS
}

pub(crate) fn recv<T: Typed>(
    buf: &mut [T],
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
    req: &mut MPI_Request,
) -> i32 {
    let src = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(rank != Context::rank(), comm, MPI_ERR_INTERN);
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_INTERN);

    let tag = Context::comm().tag_map(comm, tag);
    debug!("Recv call from {src} with tag {tag}");

    let code = Context::progress();
    if code != MPI_SUCCESS {
        return Context::err_handler().call(comm, code);
    }

    *req = Context::shm().find_unexp(src, tag);
    if !(*req).is_null() {
        let rreq = unsafe { &mut **req };
        debug!("Unexpected rank: {}, tag: {}", rreq.rank, rreq.tag);
        if rreq.cnt > buf.len() as i32 * T::into_mpi() {
            debug!("Error truncate unexpect");
            return Context::err_handler().call(comm, MPI_ERR_TRUNCATE);
        }

        unsafe {
            (buf.as_mut_ptr() as *mut c_void).copy_from(rreq.buf, rreq.cnt as usize);
            let layout = std::alloc::Layout::from_size_align_unchecked(rreq.cnt as usize, 1);
            std::alloc::dealloc(rreq.buf as *mut u8, layout);
        }
        *rreq = P_MPI_Request {
            buf: buf.as_ptr() as *mut T as *mut c_void,
            stat: MPI_Status::new(),
            comm,
            flag: 1,
            tag,
            cnt: buf.len() as i32 * p_mpi_type_size(T::into_mpi()),
            rank: src,
        };
        return MPI_SUCCESS;
    } else {
        debug!("Generate new request");
        let rreq = Context::shm().get_recv();
        if let Some(r) = rreq {
            *req = r;
            unsafe {
                **req = P_MPI_Request {
                    buf: buf.as_ptr() as *mut T as *mut c_void,
                    stat: MPI_Status::new(),
                    comm,
                    flag: 0,
                    tag,
                    cnt: buf.len() as i32 * p_mpi_type_size(T::into_mpi()),
                    rank: src,
                };
            }
            return MPI_SUCCESS;
        } else {
            *req = null_mut();
            Context::err_handler().call(comm, MPI_ERR_INTERN);
            return MPI_ERR_INTERN;
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
    let mut req: MPI_Request = uninit();
    let mut code = MPI_Isend(buf, cnt, dtype, dest, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    code = MPI_Wait(&mut req, null_mut());
    if code != MPI_SUCCESS {
        return code;
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
    let mut req: MPI_Request = uninit();
    let mut code = MPI_Irecv(buf, cnt, dtype, src, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    code = MPI_Wait(&mut req, pstat);
    if code != MPI_SUCCESS {
        return code;
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

    CHECK_RET!(MPI_Isend(sbuf, scnt, sdtype, dest, stag, comm, &mut req[0]));
    CHECK_RET!(MPI_Irecv(rbuf, rcnt, rdtype, src, rtag, comm, &mut req[1]));
    CHECK_RET!(MPI_Waitall(2, req.as_mut_ptr(), stat.as_mut_ptr()));

    unsafe {
        pstat.write(stat[1]);
    }

    MPI_SUCCESS
}

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
    let dataLen = p_mpi_type_size(dtype) * cnt;
    unsafe {
        send(
            from_raw_parts(buf as *const u8, dataLen as usize),
            dest,
            tag,
            comm,
            &mut *preq,
        )
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
    let dataLen = p_mpi_type_size(dtype) * cnt;
    unsafe {
        recv(
            from_raw_parts_mut(buf as *mut u8, dataLen as usize),
            src,
            tag,
            comm,
            &mut *preq,
        )
    }
}

#[no_mangle]
pub extern "C" fn MPI_Test(preq: *mut MPI_Request, pflag: *mut i32, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pflag.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    P_MPI_Request::test(unsafe { &mut *preq }, unsafe { &mut *pflag }, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Wait(preq: *mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    P_MPI_Request::wait(unsafe { &mut *preq }, pstat)
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
    P_MPI_Request::wait_all(
        unsafe { std::slice::from_raw_parts_mut(preq, cnt as usize) },
        unsafe { std::slice::from_raw_parts_mut(pstat, cnt as usize) },
    )
}
