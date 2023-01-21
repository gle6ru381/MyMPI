use super::keychanger::KeyChanger;
use crate::debug::DbgEntryExit;
use crate::xfer::ppp::recv::recv;
use crate::xfer::ppp::send::send;
use crate::{debug_xfer, shared::*, MPI_CHECK};

type BCastFn = fn(&mut[u8], i32, MPI_Comm) -> MpiResult;
pub const BCAST_IMPL : BCastFn = bcast_binary_tree;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const BCAST_TAG: i32 = 1;

fn bcast_binary_tree(buf: &mut [u8], mut root: i32, comm: MPI_Comm) -> MpiResult {
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);

    DbgEnEx!("Broadcast");

    if Context::size() == 1 || buf.len() == 0 {
        return Ok(());
    }

    let _ks = KeyChanger::new(Context::comm(), comm);

    let mut code: i32;
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
                code = MPI_SUCCESS;
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
