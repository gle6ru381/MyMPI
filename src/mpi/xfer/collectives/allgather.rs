use crate::{MPI_Comm, gather, bcast};
use crate::debug_xfer;
use crate::debug::DbgEntryExit;
use crate::types::MpiResult;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}


const ALLGATHER_TAG: i32 = 5;

pub fn allgather_simple(sbuf: &[u8], rbuf: &mut[u8], comm: MPI_Comm) -> MpiResult {
    DbgEnEx!("Allgather");
    
    gather::GATHER_IMPL(sbuf, rbuf, 0, comm)?;
    bcast::BCAST_IMPL(rbuf, 0, comm)?;

    Ok(())
}