use crate::context::Context;
use crate::debug::DbgEntryExit;
use crate::object::types::Typed;
use crate::xfer::request::Request;
use crate::{debug_xfer, shared::*, MPI_CHECK};
use std::ffi::c_void;
use std::ptr::null_mut;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEntryExit = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

pub(crate) fn isend<T: Typed>(
    buf: &[T],
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
    req: &mut &mut Request,
) -> MpiResult {
    DbgEnEx!("Send");

    let dest = Context::comm().rank_map(comm, rank);

    MPI_CHECK!(dest != Context::rank(), comm, MPI_ERR_INTERN)?;
    MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_TAG)?;
    let tag = Context::comm().tag_map(comm, tag);
    debug_xfer!("Send", "Send call to {dest} with tag {tag}");

    let code = Context::progress();
    if let Err(code) = code {
        return Err(Context::err_handler().call(comm, code));
    }

    let new_req = Context::shm().get_send();
    if let Some(r) = new_req {
        *req = r;
        **req = Request {
            buf: buf.as_ptr() as *mut T as *mut c_void,
            stat: MPI_Status::new(),
            comm,
            flag: 0,
            tag,
            cnt: buf.len() as i32 * type_size(T::into_mpi())?,
            rank: dest,
        };
    } else {
        *(&mut (*req as *mut Request)) = null_mut();
        Context::err_handler().call(comm, MPI_ERR_INTERN);
        return Err(MPI_ERR_INTERN);
    }
    Ok(())
}

pub(crate) fn send<T: Typed>(buf: &[T], rank: i32, tag: i32, comm: MPI_Comm) -> MpiResult {
    let mut req: &mut Request = uninit();
    isend(buf, rank, tag, comm, &mut req)?;
    req.wait(None)?;

    Ok(())
}
