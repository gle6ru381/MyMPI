use super::keychanger::KeyChanger;
use crate::debug::DbgEntryExit;
use crate::xfer::ppp::recv::recv;
use crate::xfer::ppp::send::send;
use crate::xfer::request::Request;
use crate::{debug_xfer, shared::*, MPI_CHECK};

pub type BCastFn = fn(&mut [u8], i32, MPI_Comm) -> MpiResult;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const BCAST_TAG: i32 = 1;

pub fn bcast_shm(buf: &mut [u8], root: i32, comm: MPI_Comm) -> MpiResult {
    MPI_CHECK!(
        root >= 0 && root < Context::comm_size(comm),
        comm,
        MPI_ERR_ROOT
    );

    DbgEnEx!("Broadcast");

    if Context::comm_size(comm) == 1 || buf.len() == 0 {
        return Ok(());
    }

    let _ks = KeyChanger::new(Context::comm(), comm);
    let rootRank = Context::comm_prank(comm, root);
    let rank = Context::comm_rank(comm);
    let size = Context::comm_size(comm);

    if rank == rootRank {
        let new_req = Context::shm().get_send();
        if let Some(req) = new_req {
            *req = Request {
                buf: buf.as_ptr() as *mut c_void,
                stat: MPI_Status::new(),
                comm,
                flag: 0,
                tag: BCAST_TAG,
                cnt: buf.len() as i32,
                rank: -1,
                isColl: true,
                collRoot: root,
            };
            req.wait(None)?;
            return Ok(());
        } else {
            Context::err_handler().call(comm, MPI_ERR_INTERN);
            return Err(MPI_ERR_INTERN);
        }
    } else {
        let new_req = Context::shm().get_recv();
        if let Some(req) = new_req {
            *req = Request {
                buf: buf.as_ptr() as *mut c_void,
                stat: MPI_Status::new(),
                comm,
                flag: 0,
                tag: BCAST_TAG,
                cnt: buf.len() as i32,
                rank: -1,
                isColl: true,
                collRoot: root,
            };
            req.wait(None)?;
            return Ok(());
        } else {
            Context::err_handler().call(comm, MPI_ERR_INTERN);
            return Err(MPI_ERR_INTERN);
        }
    }
}

pub fn bcast_binaty_tree(buf: &mut [u8], mut root: i32, comm: MPI_Comm) -> MpiResult {
    MPI_CHECK!(
        root >= 0 && root < Context::comm_size(comm),
        comm,
        MPI_ERR_ROOT
    );

    DbgEnEx!("Broadcast");

    if Context::comm_size(comm) == 1 || buf.len() == 0 {
        return Ok(());
    }

    let _ks = KeyChanger::new(Context::comm(), comm);

    let mut stat: MPI_Status = uninit();

    if Context::size() == 2 {
        if Context::rank() == root {
            send(buf, (root + 1) % 2, BCAST_TAG, comm)?;
        } else {
            recv(buf, root, BCAST_TAG, comm, Some(&mut stat))?;
        }
    } else {
        let mut n = 4;
        let diff = (Context::size() + Context::rank() - root) % Context::size();

        while n <= Context::size() {
            n <<= 1;
        }

        loop {
            n >>= 1;
            if n == 0 {
                break;
            }

            if Context::rank() == root {
                if diff + n < Context::size() {
                    send(
                        buf,
                        (Context::rank() + n) % Context::size(),
                        BCAST_TAG,
                        comm,
                    )?;
                }
            } else if Context::rank() == (root + n) % Context::size() {
                recv(buf, root, BCAST_TAG, comm, Some(&mut stat))?;
                root = Context::rank();
            } else if (Context::size() + Context::rank() - root) % Context::size() > n {
                root = (root + n) % Context::size();
            }
        }
    }

    Ok(())
}
