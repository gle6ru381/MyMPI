use crate::context::Context;
use crate::debug::DbgEntryExit;
use crate::object::types::Typed;
use crate::xfer::request::Request;
use crate::{debug_xfer, shared::*, MPI_CHECK};
use std::ffi::c_void;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEntryExit = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

pub(crate) fn irecv<T: Typed>(
    buf: &mut [T],
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
) -> Result<&'_ mut Request, MpiError> {
    DbgEnEx!("Recv");

    let src = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(rank != Context::rank(), comm, MPI_ERR_INTERN);
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_INTERN);

    let tag = Context::comm().tag_map(comm, tag);
    debug_xfer!("Recv", "Recv call from {src} with tag {tag}");

    let code = Context::progress();
    if let Err(code) = code {
        return Err(Context::err_handler().call(comm, code));
    }

    if let Some(r) = Context::shm().find_unexp(src, tag) {
        debug_xfer!("Recv", "Unexpected rank: {}, tag: {}", r.rank, r.tag);
        if r.cnt > buf.len() as i32 * T::into_mpi() {
            debug_xfer!("Recv", "Error truncate for unexpected data");
            return Err(Context::err_handler().call(comm, MPI_ERR_TRUNCATE));
        }

        unsafe {
            (buf.as_mut_ptr() as *mut c_void).copy_from(r.buf, r.cnt as usize);
            let layout = std::alloc::Layout::from_size_align_unchecked(r.cnt as usize, 1);
            std::alloc::dealloc(r.buf as *mut u8, layout);
        }
        *r = Request {
            buf: buf.as_ptr() as *mut T as *mut c_void,
            stat: MPI_Status::new(),
            comm,
            flag: 1,
            tag,
            cnt: buf.len() as i32 * type_size(T::into_mpi())?,
            rank: src,
        };
        return Ok(r);
    } else {
        debug_xfer!("Recv", "Create new request");
        let rreq = Context::shm().get_recv();
        if let Some(req) = rreq {
            *req = Request {
                buf: buf.as_ptr() as *mut T as *mut c_void,
                stat: MPI_Status::new(),
                comm,
                flag: 0,
                tag,
                cnt: buf.len() as i32 * type_size(T::into_mpi())?,
                rank: src,
            };
            return Ok(req);
        } else {
            Context::err_handler().call(comm, MPI_ERR_INTERN);
            return Err(MPI_ERR_INTERN);
        }
    }
}

pub(crate) fn recv<T: Typed>(
    buf: &mut [T],
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
    pstat: Option<&mut MPI_Status>,
) -> MpiResult {
    let req = irecv(buf, rank, tag, comm)?;
    debug_xfer!("Recv", "Irecv create request: {}", req.rank);
    req.wait(pstat)?;

    Ok(())
}
