pub (crate) mod bcast;
pub (crate) mod gather;
mod keychanger;

use crate::debug::DbgEntryExit;
use crate::{
    debug_xfer, shared::*, MPI_Comm_call_errhandler, MPI_Comm_rank, MPI_Comm_size,
    MPI_Type_size, MPI_CHECK, MPI_CHECK_OP, check_type,
};
use crate::bindings::{MPI_Recv, MPI_Send, MPI_Sendrecv};
use self::bcast::BCAST_IMPL;

use super::reducefunc::*;
use std::slice::from_raw_parts_mut;
use std::{alloc::alloc, alloc::dealloc, alloc::Layout, ffi::c_void};

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEnEx = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

const BARRIER_TAG: i32 = 0;
const REDUCE_TAG: i32 = 2;
const ALLREDUCE_TAG: i32 = 4;

type FUNC = fn(*const c_void, *mut c_void, i32, i32);

const FUNCTIONS: [FUNC; 3] = [max, min, sum];

pub(crate) fn p_mpi_check_op(op: MPI_Op, comm: MPI_Comm) -> i32 {
    todo!()
    // MPI_CHECK_RET!(
    //     op == MPI_MAX || op == MPI_MIN || op == MPI_SUM,
    //     comm,
    //     MPI_ERR_OP
    // )
}

#[no_mangle]
pub extern "C" fn MPI_Barrier(comm: MPI_Comm) -> i32 {
    let mut size: i32 = uninit();
    let mut rank: i32 = uninit();

    DbgEnEx!("Barrier");

    CHECK_RET!(MPI_Comm_size(comm, &mut size));
    CHECK_RET!(MPI_Comm_rank(comm, &mut rank));

    if size == 1 {
        return MPI_SUCCESS;
    }

    let cg = Context::comm();

    cg.inc_key(comm);

    let mut buf: i8 = uninit();
    let mut stat: MPI_Status = uninit();
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
            CHECK_RET!(MPI_Send(
                pbuf,
                0,
                MPI_BYTE,
                (rank + 1) % size,
                BARRIER_TAG,
                comm
            ));
            CHECK_RET!(MPI_Recv(
                pbuf,
                0,
                MPI_BYTE,
                (size + rank - 1) % size,
                BARRIER_TAG,
                comm,
                &mut stat
            ));
        } else {
            CHECK_RET!(MPI_Recv(
                pbuf,
                0,
                MPI_BYTE,
                (size + rank - 1) % size,
                BARRIER_TAG,
                comm,
                &mut stat
            ));
            CHECK_RET!(MPI_Send(
                pbuf,
                0,
                MPI_BYTE,
                (rank + 1) % size,
                BARRIER_TAG,
                comm
            ));
        }
    }

    cg.dec_key(comm);

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Bcast(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    let dataLen = type_size(dtype).unwrap() * cnt;
    if let Err(code) = BCAST_IMPL(unsafe {from_raw_parts_mut(buf as *mut u8, dataLen as usize)}, root, comm) {
        return code as i32;
    }
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Reduce(
    sbuf: *const c_void,
    mut rbuf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    op: MPI_Op,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    MPI_CHECK_OP!(op, comm);
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);
    MPI_CHECK!(!rbuf.is_null() && !sbuf.is_null(), comm, MPI_ERR_BUFFER);
    MPI_CHECK!(cnt >= 0, comm, MPI_ERR_COUNT);
    check_type(dtype, comm);

    DbgEnEx!("Reduce");

    if cnt == 0 {
        return MPI_SUCCESS;
    }

    if Context::size() == 1 {
        // if p_mpi_type_copy(rbuf, sbuf, cnt, dtype).is_null() {
        //     MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
        //     return MPI_ERR_INTERN;
        // }
        return MPI_SUCCESS;
    }

    let cg = Context::comm();
    cg.inc_key(comm);
    let mut code: i32;
    let mut stat: MPI_Status = uninit();

    if Context::size() == 2 {
        if Context::rank() == root {
            code = MPI_Recv(
                rbuf,
                cnt,
                dtype,
                (root + 1) % 2,
                REDUCE_TAG,
                comm,
                &mut stat,
            );
            if code == MPI_SUCCESS {
                FUNCTIONS[op as usize](sbuf, rbuf, cnt, dtype);
            }
        } else {
            code = MPI_Send(sbuf, cnt, dtype, root, REDUCE_TAG, comm);
        }
    } else {
        let mut ext = uninit();
        let diff = (Context::size() + Context::rank() - root) % Context::size();

        code = MPI_Type_size(dtype, &mut ext);
        if code != MPI_SUCCESS {
            cg.dec_key(comm);
            return code;
        }

        let layout = unsafe { Layout::from_size_align_unchecked((ext * cnt) as usize, 1) };
        if Context::rank() != root {
            rbuf = unsafe { std::alloc::alloc(layout) as *mut c_void };
            if rbuf.is_null() {
                cg.dec_key(comm);
                MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                return -1;//MPI_ERR_INTERN;
            }
        }

        if diff % 2 != 0 {
            code = MPI_Send(
                sbuf,
                cnt,
                dtype,
                (Context::size() + Context::rank() - 1) % Context::size(),
                REDUCE_TAG,
                comm,
            );
        } else if diff < (Context::size() - 1) {
            code = MPI_Recv(
                rbuf,
                cnt,
                dtype,
                (Context::rank() + 1) % Context::size(),
                REDUCE_TAG,
                comm,
                &mut stat,
            );
            if code == MPI_SUCCESS {
                FUNCTIONS[op as usize](sbuf, rbuf, cnt, dtype);
            }
        }

        if code == MPI_SUCCESS {
            let mut tbuf: *mut c_void = null_mut();

            if diff % 4 != 0 {
                if diff % 2 == 0 {
                    if diff < Context::size() - 1 {
                        code = MPI_Send(
                            rbuf,
                            cnt,
                            dtype,
                            (Context::size() + Context::rank() - 2) % Context::size(),
                            REDUCE_TAG,
                            comm,
                        );
                    } else {
                        code = MPI_Send(
                            sbuf,
                            cnt,
                            dtype,
                            (Context::size() + Context::rank() - 2) % Context::size(),
                            REDUCE_TAG,
                            comm,
                        );
                    }
                }
            } else if diff < Context::size() - 2 {
                tbuf = unsafe { std::alloc::alloc(layout) as *mut c_void };
                if tbuf.is_null() {
                    if Context::rank() != root {
                        unsafe {
                            std::alloc::dealloc(rbuf as *mut u8, layout);
                        }
                    }
                    cg.dec_key(comm);
                    MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                    return -1//MPI_ERR_INTERN;
                }

                code = MPI_Recv(
                    tbuf,
                    cnt,
                    dtype,
                    (Context::rank() + 2) % Context::size(),
                    REDUCE_TAG,
                    comm,
                    &mut stat,
                );
                if code == MPI_SUCCESS {
                    debug_xfer!("Reduce", "Evaluate function");
                    FUNCTIONS[op as usize](tbuf, rbuf, cnt, dtype);
                }
            }

            if code == MPI_SUCCESS {
                let mut i = 8;
                let mut iold = 4;
                while iold < Context::size() {
                    if diff % i != 0 {
                        if diff % iold == 0 {
                            if diff < Context::size() - 1 {
                                code = MPI_Send(
                                    rbuf,
                                    cnt,
                                    dtype,
                                    (Context::size() + Context::rank() - iold) % Context::size(),
                                    REDUCE_TAG,
                                    comm,
                                );
                            } else {
                                code = MPI_Send(
                                    sbuf,
                                    cnt,
                                    dtype,
                                    (Context::size() + Context::rank() - iold) % Context::size(),
                                    REDUCE_TAG,
                                    comm,
                                );
                            }
                        }
                    } else if diff < Context::size() - iold {
                        code = MPI_Recv(
                            tbuf,
                            cnt,
                            dtype,
                            (Context::rank() + iold) % Context::size(),
                            REDUCE_TAG,
                            comm,
                            &mut stat,
                        );
                        if code == MPI_SUCCESS {
                            FUNCTIONS[op as usize](tbuf, rbuf, cnt, dtype);
                        }
                    }

                    if code != MPI_SUCCESS {
                        break;
                    }
                    iold = i;
                    i <<= 1;
                }
            }

            if !tbuf.is_null() {
                unsafe {
                    std::alloc::dealloc(tbuf as *mut u8, layout);
                }
            }
        }

        if Context::rank() != root {
            unsafe {
                std::alloc::dealloc(rbuf as *mut u8, layout);
            }
        }
    }

    cg.dec_key(comm);

    code
}

#[no_mangle]
pub extern "C" fn MPI_Allreduce(
    sbuf: *const c_void,
    rbuf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    op: MPI_Op,
    comm: MPI_Comm,
) -> i32 {
    MPI_CHECK!(!sbuf.is_null() && !rbuf.is_null(), comm, MPI_ERR_BUFFER);
    check_type(dtype, comm);
    MPI_CHECK_OP!(op, comm);

    DbgEnEx!("Allreduce");

    let mut n = 1;
    while n <= Context::size() {
        n <<= 1;
    }
    n >>= 1;
    if n == Context::size() && false {
        let mut stat = MPI_Status::uninit();

        if cnt == 0 {
            return MPI_SUCCESS;
        }

        if Context::size() == 1 {
            // if p_mpi_type_copy(rbuf, sbuf, cnt, dtype).is_null() {
            //     MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
            //     return MPI_ERR_INTERN;
            // }
            return MPI_SUCCESS;
        }

        let c = Context::comm();
        c.inc_key(comm);

        let mut code = MPI_Sendrecv(
            sbuf,
            cnt,
            dtype,
            Context::rank() ^ 1,
            ALLREDUCE_TAG,
            rbuf,
            cnt,
            dtype,
            Context::rank() ^ 1,
            ALLREDUCE_TAG,
            comm,
            &mut stat,
        );
        if code == MPI_SUCCESS {
            FUNCTIONS[op as usize](sbuf, rbuf, cnt, dtype);
            if Context::size() > 2 {
                let mut i = 2;
                let ext = type_size(dtype).unwrap();
                let tbuf: *mut c_void;

                let layout = unsafe { Layout::from_size_align_unchecked((ext * cnt) as usize, 1) };
                tbuf = unsafe { alloc(layout) } as *mut c_void;
                if tbuf.is_null() {
                    c.dec_key(comm);
                    Context::call_error(comm, MPI_ERR_INTERN);
                    return -1;//MPI_ERR_INTERN;
                }

                while i <= n {
                    code = MPI_Sendrecv(
                        rbuf,
                        cnt,
                        dtype,
                        Context::rank() ^ i,
                        ALLREDUCE_TAG,
                        tbuf,
                        cnt,
                        dtype,
                        Context::rank() ^ i,
                        ALLREDUCE_TAG,
                        comm,
                        &mut stat,
                    );
                    if code == MPI_SUCCESS {
                        FUNCTIONS[op as usize](tbuf, rbuf, cnt, dtype);
                    } else {
                        break;
                    }
                    i <<= 1;
                }
                unsafe { dealloc(tbuf as *mut u8, layout) };
            }
        }
        c.dec_key(comm);
    } else {
        CHECK_RET!(MPI_Reduce(sbuf, rbuf, cnt, dtype, op, 0, comm));
        CHECK_RET!(MPI_Bcast(rbuf, cnt, dtype, 0, comm));
    }
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Gather(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    mut rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    root: i32,
    comm: MPI_Comm,
) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Allgather(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    comm: MPI_Comm,
) -> i32 {
    DbgEnEx!("Allgather");

    CHECK_RET!(MPI_Gather(sbuf, scnt, sdtype, rbuf, rcnt, rdtype, 0, comm));
    CHECK_RET!(MPI_Bcast(rbuf, rcnt * Context::size(), rdtype, 0, comm));
    MPI_SUCCESS
}
