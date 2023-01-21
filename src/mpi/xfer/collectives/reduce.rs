use crate::buffer::DynBuffer;
use crate::debug::DbgEntryExit;
use crate::xfer::ppp::recv::recv;
use crate::xfer::ppp::send::send;
use crate::{debug_xfer, MPI_Comm, MpiResult, MPI_Op, MPI_CHECK, MPI_MAX, MPI_MIN, MPI_SUM, MPI_Datatype, check_type, type_size};
use crate::context::Context;
use super::keychanger::KeyChanger;
use crate::types::MpiError::*;
use super::reducefunc::*;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const REDUCE_TAG: i32 = 3;

type ReduceFn = fn(&[u8], &mut[u8], MPI_Datatype, MPI_Op, i32, MPI_Comm) -> MpiResult;
pub const REDUCE_IMPL: ReduceFn = reduce_ring;

type FUNC = fn(*const u8, *mut u8, i32, i32);

pub (super) const FUNCTIONS: [FUNC; 3] = [max, min, sum];

pub (super) fn check_op(op: MPI_Op, comm: MPI_Comm) -> MpiResult {
    MPI_CHECK!(matches!(op, MPI_MAX | MPI_MIN | MPI_SUM), comm, MPI_ERR_OP)
}

pub fn reduce_ring(sbuf: &[u8], rbuf: &mut [u8], dtype: MPI_Datatype, op: MPI_Op, root: i32, comm: MPI_Comm) -> MpiResult {
    check_op(op, comm)?;
    check_type(dtype, comm)?;

    DbgEnEx!("Reduce");

    let size = Context::comm_size(comm);
    let rank = Context::comm_rank(comm);
    let blk_size = sbuf.len() / type_size(dtype)? as usize;

    if blk_size == 0 {
        return Ok(());
    }

    if size == 1 {
        todo!()
    }

    let _kc = KeyChanger::new(Context::comm(), comm);

    if size == 2 {
        if rank == root {
            recv(rbuf, (root + 1) % 2, REDUCE_TAG, comm, None)?;
            FUNCTIONS[op as usize](sbuf.as_ptr(), rbuf.as_mut_ptr(), blk_size as i32, dtype);
        } else {
            send(sbuf, root, REDUCE_TAG, comm)?;
        }
        return Ok(());
    }

    let diff = (size + rank - root) % size;

    let buff: &mut [u8];
    let _dyn_buffer: DynBuffer;
    if rank != root {
        _dyn_buffer = DynBuffer::new(sbuf.len());
        buff = _dyn_buffer.to_slice();
    } else {
        _dyn_buffer = DynBuffer::empty();
        buff = rbuf;
    }

    if diff % 2 != 0 {
        send(sbuf, (size + rank - 1) % size, REDUCE_TAG, comm)?;
    } else if diff < size - 1 {
        recv(buff, (rank + 1) % size, REDUCE_TAG, comm, None)?;
        FUNCTIONS[op as usize](sbuf.as_ptr(), buff.as_mut_ptr(), blk_size as i32, dtype);
    }

    let tbuf: DynBuffer;
    if diff % 4 != 0 {
        if diff % 2 == 0 {
            if diff < size - 1 {
                send(buff, (size + rank - 2) % size, REDUCE_TAG, comm)?;
            } else {
                send(sbuf, (size + rank - 2) % size, REDUCE_TAG, comm)?;
            }
        }
        tbuf = DynBuffer::empty();
    } else if diff < size - 2 {
        tbuf = DynBuffer::new(sbuf.len());
        
        recv(tbuf.to_slice(), (rank + 2) % size, REDUCE_TAG, comm, None)?;
        FUNCTIONS[op as usize](tbuf.to_slice().as_ptr(), buff.as_mut_ptr(), blk_size as i32, dtype);
    } else {
        tbuf = DynBuffer::empty();
    }

    let mut i = 8;
    let mut iold = 4;
    while iold < size {
        if diff % i != 0 {
            if diff % iold == 0 {
                if diff < size - 1 {
                    send(buff, (size + rank - iold) % size, REDUCE_TAG, comm)?;
                } else {
                    send(sbuf, (size + rank - iold) % size, REDUCE_TAG, comm)?;
                }
            }
        } else if diff < size - iold {
            recv(tbuf.to_slice(), (rank + iold) % size, REDUCE_TAG, comm, None)?;
            FUNCTIONS[op as usize](tbuf.to_slice().as_ptr(), buff.as_mut_ptr(), blk_size as i32, dtype);
        }

        iold = i;
        i <<= 1;
    }

    Ok(())
}