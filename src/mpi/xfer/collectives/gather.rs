use super::keychanger::KeyChanger;
use crate::backend::memory::memcpy_slice;
use crate::debug::DbgEntryExit;
use crate::xfer::ppp::recv::recv;
use crate::xfer::ppp::send::send;
use crate::{debug_xfer, shared::*, MPI_CHECK};

pub type GatherFn = fn(&[u8], &mut [u8], i32, MPI_Comm) -> MpiResult;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const GATHER_TAG: i32 = 2;

pub fn gather_ring(sbuf: &[u8], rbuf: &mut [u8], root: i32, comm: MPI_Comm) -> MpiResult {
    DbgEnEx!("Gather");

    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER)?;
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT)?;

    let csize = Context::comm_size(comm);
    let rank = Context::comm_rank(comm);
    let blk_size = rbuf.len() / csize as usize;

    if sbuf.len() == 0 {
        return Ok(());
    }

    if csize == 1 {
        memcpy_slice(rbuf, sbuf, rbuf.len());
        return Ok(());
    }

    let _kc = KeyChanger::new(Context::comm(), comm);

    let mut stat: MPI_Status = uninit();

    if csize == 2 {
        if rank == root {
            let offset = blk_size * ((root as usize + 1) % 2);
            recv(
                &mut rbuf[offset..offset + blk_size],
                (root + 1) % 2,
                GATHER_TAG,
                comm,
                Some(&mut stat),
            )?;
            memcpy_slice(&mut rbuf[blk_size * root as usize..], &sbuf, blk_size);
        } else {
            send(&sbuf[..blk_size], root, GATHER_TAG, comm)?;
        }
    } else {
        if Context::rank() == root {
            memcpy_slice(
                &mut rbuf[blk_size * Context::rank() as usize..],
                sbuf,
                blk_size,
            );
            send(rbuf, (rank + 1) % csize, GATHER_TAG, comm)?;
            recv(
                rbuf,
                (csize + rank - 1) % csize,
                GATHER_TAG,
                comm,
                Some(&mut stat),
            )?;
        } else {
            let buf = crate::buffer::DynBuffer::new(rbuf.len());
            recv(
                buf.to_slice(),
                (csize + rank - 1) % csize,
                GATHER_TAG,
                comm,
                Some(&mut stat),
            )?;
            memcpy_slice(
                &mut buf.to_slice()[blk_size * rank as usize..],
                sbuf,
                blk_size,
            );
            send(buf.to_slice(), (rank + 1) % csize, GATHER_TAG, comm)?;
        }
    }

    debug_xfer!("Gather", "Gather done: {:?}", rbuf);

    Ok(())
}
