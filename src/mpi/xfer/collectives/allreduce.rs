use crate::backend::memory::{memcpy_slice};
use crate::buffer::DynBuffer;
use crate::debug::DbgEntryExit;
use crate::xfer::ppp::sendrecv;
use crate::{debug_xfer, MPI_Comm, MpiResult, MPI_Op, MPI_Datatype, check_type};
use crate::context::Context;
use super::keychanger::KeyChanger;
use super::reduce::FUNCTIONS;
use super::reduce::check_op;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const ALLREDUCE_TAG: i32 = 4;

pub fn allreduce_hypercube(sbuf: &[u8], rbuf: &mut[u8], dtype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> MpiResult {
    check_type(dtype, comm)?;
    check_op(op, comm)?;

    DbgEnEx!("Allreduce");

    let size = Context::comm_size(comm);
    let rank = Context::comm_rank(comm);
    let blk_size = sbuf.len() / size as usize;
    let mut n = 1;

    if sbuf.len() == 0 {
        return Ok(());
    }

    while n <= size {
        n <<= 1;
    }
    n >>= 1;

    if size == 1 {
        memcpy_slice(rbuf, sbuf, sbuf.len());
        return Ok(());
    }

    if n == size {
        let _kc = KeyChanger::new(Context::comm(), comm);

        sendrecv(sbuf, rank ^ 1, ALLREDUCE_TAG, rbuf, rank ^ 1, ALLREDUCE_TAG, comm)?;
        FUNCTIONS[op as usize](sbuf.as_ptr(), rbuf.as_mut_ptr(), blk_size as i32, dtype);
        
        let mut i = 2;
        let tbuf = DynBuffer::new(sbuf.len());

        while i <= n {
            sendrecv(rbuf, rank ^ i, ALLREDUCE_TAG, tbuf.to_slice(), rank ^ i, ALLREDUCE_TAG, comm)?;
            FUNCTIONS[op as usize](tbuf.to_slice().as_ptr(), rbuf.as_mut_ptr(), blk_size as i32, dtype);

            i <<= 1;
        }
    }

    Ok(())
}