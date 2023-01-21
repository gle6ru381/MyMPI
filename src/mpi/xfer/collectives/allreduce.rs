use crate::backend::memory::memcpy_slice;
use crate::bcast::BCAST_IMPL;
use crate::buffer::DynBuffer;
use crate::debug::DbgEntryExit;
use crate::reduce::REDUCE_IMPL;
use crate::xfer::ppp::sendrecv;
use crate::{debug_coll, MPI_Comm, MpiResult, MPI_Op, MPI_Datatype, check_type};
use crate::context::Context;
use super::keychanger::KeyChanger;
use super::reduce::FUNCTIONS;
use super::reduce::check_op;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_coll!($name, "{s}"));
    };
}

type AllreduceFn = fn(&[u8], &mut[u8], MPI_Datatype, MPI_Op, MPI_Comm) -> MpiResult;
pub const ALLREDUCE_IMPL: AllreduceFn = allreduce_simple;

const ALLREDUCE_TAG: i32 = 4;

pub fn allreduce_simple(sbuf: &[u8], rbuf: &mut[u8], dtype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> MpiResult {
    DbgEnEx!("Allreduce");

    REDUCE_IMPL(sbuf, rbuf, dtype, op, 0, comm)?;
    BCAST_IMPL(rbuf, 0, comm)?;

    Ok(())
}

pub fn allreduce_tree(sbuf: &[u8], rbuf: &mut[u8], dtype: MPI_Datatype, op: MPI_Op, comm: MPI_Comm) -> MpiResult {
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

    debug_coll!("Allreduce", "N: {n}, size: {size}");

    if n == size {
        let _kc = KeyChanger::new(Context::comm(), comm);

        debug_coll!("Allreduce", "Sendrecv to: {}", rank ^ 1);
        sendrecv(sbuf, rank ^ 1, ALLREDUCE_TAG, rbuf, rank ^ 1, ALLREDUCE_TAG, comm)?;
        FUNCTIONS[op as usize](sbuf.as_ptr(), rbuf.as_mut_ptr(), blk_size as i32, dtype);
        
        let mut i = 2;
        let tbuf = DynBuffer::new(sbuf.len());

        while i < n {
            debug_coll!("Allreduce", "Sendrecv to: {}", rank ^ i);
            sendrecv(rbuf, rank ^ i, ALLREDUCE_TAG, tbuf.to_slice(), rank ^ i, ALLREDUCE_TAG, comm)?;
            FUNCTIONS[op as usize](tbuf.to_slice().as_ptr(), rbuf.as_mut_ptr(), blk_size as i32, dtype);

            i <<= 1;
        }
    }

    Ok(())
}