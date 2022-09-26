use crate::{types::*, private::*, MPI_CHECK, MPI_Comm_size, MPI_Comm_rank, MPI_Send, MPI_Recv, uninit, MPI_CHECK_OP, MPI_CHECK_TYPE, MPI_Comm_call_errhandler};
use std::ffi::c_void;
use crate::reducefuc::*;

const BARRIER_TAG : i32 = 0;
const BCAST_TAG : i32 = 1;
const READUCE_TAG : i32 = 2;
const GATHER_TAG : i32 = 3;

type FUNC = fn(*const c_void, *mut c_void, i32, i32);

const FUNCTIONS : [FUNC; 3] = [max, min, sum];

pub (crate) fn p_mpi_check_op(op : MPI_Op, comm : MPI_Comm) -> i32 {
    MPI_CHECK!(op == MPI_MAX || op == MPI_MIN || op == MPI_SUM, comm, MPI_ERR_OP)
}

fn p_mpi_type_copy(dest : *mut c_void, src : *const c_void, len : i32, dtype : MPI_Datatype) -> *mut c_void {
    debug_assert!(Context::is_init());
    debug_assert!(!dest.is_null() && !src.is_null());
    debug_assert!(len >= 0);
    MPI_CHECK_TYPE!(dtype, MPI_COMM_WORLD);

    unsafe {
        dest.copy_from(src, (len * p_mpi_type_size(dtype)) as usize);
        dest
    }
}

fn p_mpi_type_copy2(dest : *mut c_void, dcnt : i32, ddtype : MPI_Datatype, src : *const c_void, scnt : i32, sdtype : MPI_Datatype) -> *mut c_void
{
    debug_assert!(Context::is_init());
    debug_assert!(!dest.is_null() && !src.is_null());
    debug_assert!(dcnt >= 0 && scnt >= 0);
    MPI_CHECK_TYPE!(ddtype, MPI_COMM_WORLD);
    MPI_CHECK_TYPE!(sdtype, MPI_COMM_WORLD);
    debug_assert!(dcnt == scnt);
    debug_assert!(sdtype == ddtype);

    unsafe {
        dest.copy_from(src, (scnt * p_mpi_type_size(sdtype)) as usize);
        dest
    }
}

#[no_mangle]
pub extern "C" fn MPI_Barrier(comm : MPI_Comm) -> i32 {
    let mut size : i32 = uninit!();
    let mut rank : i32 = uninit!();

    CHECK_RET!(MPI_Comm_size(comm, &mut size));
    CHECK_RET!(MPI_Comm_rank(comm, &mut rank));

    if size == 1 {
        return MPI_SUCCESS;
    }

    p_mpi_inc_key(comm);

    let mut buf : i8 = uninit!();
    let mut stat : MPI_Status = uninit!();
    let pbuf = &mut buf as *mut i8 as *mut c_void;

    if size == 2 {
        if rank == 0 {
            CHECK_RET!(MPI_Send(pbuf, 0, MPI_BYTE, 1, BARRIER_TAG, comm));
            CHECK_RET!(MPI_Recv(pbuf, 0, MPI_BYTE, 1, BARRIER_TAG, comm, &mut stat));
        } else {
            CHECK_RET!(MPI_Recv(pbuf, 0, MPI_BYTE, 0, BARRIER_TAG, comm, &mut stat));
            CHECK_RET!(MPI_Send(pbuf, 0, MPI_BYTE, 0, BARRIER_TAG, comm));
        }
    } else {
        if rank == 0 {
            CHECK_RET!(MPI_Send(pbuf, 0, MPI_BYTE, (rank + 1) % size, BARRIER_TAG, comm));
            CHECK_RET!(MPI_Recv(pbuf, 0, MPI_BYTE, (size + rank - 1) % size, BARRIER_TAG, comm, &mut stat));
        } else {
            CHECK_RET!(MPI_Recv(pbuf, 0, MPI_BYTE, (size + rank - 1) % size, BARRIER_TAG, comm, &mut stat));
            CHECK_RET!(MPI_Send(pbuf, 0, MPI_BYTE, (rank + 1) % size, BARRIER_TAG, comm));
        }
    }

    p_mpi_dec_key(comm);

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Bcast(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, root : i32, comm : MPI_Comm) -> i32 {
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);
    MPI_CHECK!(Context::size() <= 2, comm, MPI_ERR_INTERN);

    if Context::size() == 1 {
        return MPI_SUCCESS;
    }

    debug!("Begin Broadcast");

    p_mpi_inc_key(comm);
    let code : i32;
    let mut stat : MPI_Status = uninit!();
    if Context::rank() == root {
        code = MPI_Send(buf, cnt, dtype, (root + 1) % 2, BCAST_TAG, comm);
    } else {
        code = MPI_Recv(buf, cnt, dtype, root, BCAST_TAG, comm, &mut stat);
    }

    p_mpi_dec_key(comm);
    code
}

#[no_mangle]
pub extern "C" fn MPI_Reduce(sbuf : *const c_void, rbuf : *mut c_void, cnt : i32, dtype : MPI_Datatype, op : MPI_Op, root : i32, comm : MPI_Comm) -> i32 {
    MPI_CHECK_OP!(op, comm);
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);
    MPI_CHECK!(Context::size() <= 2, comm, MPI_ERR_INTERN);

    if Context::size() == 1 {
        MPI_CHECK!(!rbuf.is_null() && !sbuf.is_null(), comm, MPI_ERR_BUFFER);
        MPI_CHECK!(cnt >= 0, comm, MPI_ERR_COUNT);
        MPI_CHECK_TYPE!(dtype, comm);

        p_mpi_type_copy(rbuf, sbuf, cnt, dtype);
        return MPI_SUCCESS;
    }

    p_mpi_inc_key(comm);
    let code : i32;
    let mut stat : MPI_Status = uninit!();
    if Context::rank() == root {
        code = MPI_Recv(rbuf, cnt, dtype, (root + 1) % 2, READUCE_TAG, comm, &mut stat);
        if code == MPI_SUCCESS {
            FUNCTIONS[op as usize](sbuf, rbuf, cnt, dtype);
        }
    } else {
        code = MPI_Send(sbuf, cnt, dtype, root, READUCE_TAG, comm);
    }

    p_mpi_dec_key(comm);

    code
}

#[no_mangle]
pub extern "C" fn MPI_Allreduce(sbuf : *const c_void, rbuf : *mut c_void, cnt : i32, dtype : MPI_Datatype, op : MPI_Op, comm : MPI_Comm) -> i32 {
    CHECK_RET!(MPI_Reduce(sbuf, rbuf, cnt, dtype, op, 0, comm));
    CHECK_RET!(MPI_Bcast(rbuf, cnt, dtype, 0, comm));
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Gather(sbuf : *const c_void, scnt : i32, sdtype : MPI_Datatype, rbuf : *mut c_void, rcnt : i32, rdtype : MPI_Datatype, root : i32, comm : MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);

    if Context::size() == 1 {
        MPI_CHECK!(!sbuf.is_null() && !rbuf.is_null(), comm, MPI_ERR_BUFFER);
        MPI_CHECK!(scnt >= 0 && rcnt >= 0, comm, MPI_ERR_COUNT);
        MPI_CHECK_TYPE!(sdtype, comm);
        MPI_CHECK_TYPE!(rdtype, comm);

        if p_mpi_type_copy2(rbuf, rcnt, rdtype, sbuf, scnt, sdtype).is_null() {
            MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
            return MPI_ERR_INTERN;
        }
        return MPI_SUCCESS
    } else if Context::size() == 2 {
        MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
        return MPI_ERR_INTERN;
    }

    p_mpi_inc_key(comm);

    let mut stat : MPI_Status = uninit!();
    let mut code;

    if Context::rank() == root {
        let tsize = p_mpi_type_size(rdtype) * rcnt;
        let dst = (root + 1) % 2;
        unsafe {
            code = MPI_Recv(rbuf.add((tsize * dst) as usize), rcnt, rdtype, dst, GATHER_TAG, comm, &mut stat);
            if code == MPI_SUCCESS {
                if p_mpi_type_copy2(rbuf.add((tsize * root) as usize), rcnt, rdtype, sbuf, scnt, sdtype).is_null() {
                    p_mpi_dec_key(comm);
                    MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                    return MPI_ERR_INTERN;
                }
            }
        }
    } else {
        code = MPI_Send(sbuf, scnt, sdtype, root, GATHER_TAG, comm);
    }
    p_mpi_dec_key(comm);
    code
}

#[no_mangle]
pub extern "C" fn MPI_Allgather(sbuf : *const c_void, scnt : i32, sdtype : MPI_Datatype, rbuf : *mut c_void, rcnt : i32, rdtype : MPI_Datatype, comm : MPI_Comm) -> i32 {
    CHECK_RET!(MPI_Gather(sbuf, scnt, sdtype, rbuf, rcnt, rdtype, 0, comm));
    CHECK_RET!(MPI_Bcast(rbuf, rcnt * Context::size(), rdtype, 0, comm));
    MPI_SUCCESS
}