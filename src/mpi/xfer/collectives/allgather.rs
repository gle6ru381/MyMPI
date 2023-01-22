use crate::context::Context;
use crate::debug::DbgEntryExit;
use crate::debug_xfer;
use crate::types::MpiResult;
use crate::MPI_Comm;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

pub type AllgatherFn = fn(&[u8], &mut [u8], MPI_Comm) -> MpiResult;

#[allow(dead_code)]
const ALLGATHER_TAG: i32 = 5;

pub fn allgather_simple(sbuf: &[u8], rbuf: &mut [u8], comm: MPI_Comm) -> MpiResult {
    DbgEnEx!("Allgather");

    Context::gather()(sbuf, rbuf, 0, comm)?;
    Context::bcast()(rbuf, 0, comm)?;

    Ok(())
}
