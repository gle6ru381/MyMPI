use crate::context::Context;
use crate::debug::DbgEntryExit;
use crate::object::types::Typed;
use crate::{debug_xfer, shared::*, MPI_CHECK};
use std::ffi::c_void;
use std::ptr::null_mut;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEntryExit = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

impl P_MPI_Request {
    pub fn test(preq: &mut MPI_Request, pflag: &mut i32, pstat: *mut MPI_Status) -> i32 {
        DbgEnEx!("Test");

        let code = Context::progress();
        if code != MPI_SUCCESS {
            return Context::err_handler().call(MPI_COMM_WORLD, code);
        }
        *pflag = 0;
        if !(*preq).is_null() {
            let req = unsafe { &mut **preq };
            if req.flag != 0 {
                debug_xfer!(
                    "Test",
                    "Find request with tag: {}, rank: {}",
                    req.tag,
                    req.rank
                );
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
        DbgEnEx!("Wait");
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
        DbgEnEx!("WaitAll");
        debug_assert!(reqs.len() == pstat.len());

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
    DbgEnEx!("Send");

    let dest = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(dest != Context::rank(), comm, MPI_ERR_INTERN);
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_TAG);

    let tag = Context::comm().tag_map(comm, tag);
    debug_xfer!("Send", "Send call to {dest} with tag {tag}");

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
    DbgEnEx!("Recv");

    let src = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(rank != Context::rank(), comm, MPI_ERR_INTERN);
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_INTERN);

    let tag = Context::comm().tag_map(comm, tag);
    debug_xfer!("Recv", "Recv call from {src} with tag {tag}");

    let code = Context::progress();
    if code != MPI_SUCCESS {
        return Context::err_handler().call(comm, code);
    }

    *req = Context::shm().find_unexp(src, tag);
    if !(*req).is_null() {
        let rreq = unsafe { &mut **req };
        debug_xfer!("Recv", "Unexpected rank: {}, tag: {}", rreq.rank, rreq.tag);
        if rreq.cnt > buf.len() as i32 * T::into_mpi() {
            debug_xfer!("Recv", "Error truncate for unexpected data");
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
        debug_xfer!("Recv", "Create new request");
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