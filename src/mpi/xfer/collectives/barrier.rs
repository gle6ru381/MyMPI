use crate::debug::DbgEntryExit;
use crate::xfer::ppp::recv::recv;
use crate::xfer::ppp::send::send;
use crate::{debug_xfer, MPI_Comm, MpiResult};
use crate::context::Context;
use super::keychanger::KeyChanger;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

type BarrierFn = fn(MPI_Comm) -> MpiResult;
pub const BARRIER_IMPL: BarrierFn = barrier_simple;

const BARRIER_TAG: i32 = 3;

pub fn barrier_simple(comm: MPI_Comm) -> MpiResult {
    let rank = Context::comm_rank(comm);
    let size = Context::comm_size(comm);

    DbgEnEx!("Barrier");

    if size == 1 {
        return Ok(());
    }

    let _kc = KeyChanger::new(Context::comm(), comm);

    if size == 2 {
        if rank == 0 {
            send(&[0;0], 1, BARRIER_TAG, comm)?;
            recv(&mut[0;0], 1, BARRIER_TAG, comm, None)?;
        } else {
            recv(&mut [0;0], 0, BARRIER_TAG, comm, None)?;
            send(&[0;0], 0, BARRIER_TAG, comm)?;
        }
        return Ok(());
    }

    if rank == 0 {
        send(&[0;0], (rank + 1) % size, BARRIER_TAG, comm)?;
        recv(&mut [0;0], (size + rank - 1) % size, BARRIER_TAG, comm, None)?;
    } else {
        recv(&mut [0;0], (size + rank - 1) % size, BARRIER_TAG, comm, None)?;
        send(&[0;0], (rank + 1) % size, BARRIER_TAG, comm)?;
    }

    Ok(())
}