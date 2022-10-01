use crate::reducefuc::*;
use crate::{
    private::*, types::*, MPI_Comm_call_errhandler, MPI_Comm_rank, MPI_Comm_size, MPI_Recv,
    MPI_Send, MPI_Type_size, MPI_CHECK, MPI_CHECK_OP, MPI_CHECK_TYPE,
};
use std::{alloc::Layout, ffi::c_void};

const BARRIER_TAG: i32 = 0;
const BCAST_TAG: i32 = 1;
const REDUCE_TAG: i32 = 2;
const GATHER_TAG: i32 = 3;

type FUNC = fn(*const c_void, *mut c_void, i32, i32);

const FUNCTIONS: [FUNC; 3] = [max, min, sum];

pub(crate) fn p_mpi_check_op(op: MPI_Op, comm: MPI_Comm) -> i32 {
    MPI_CHECK!(
        op == MPI_MAX || op == MPI_MIN || op == MPI_SUM,
        comm,
        MPI_ERR_OP
    )
}

fn p_mpi_type_copy(
    dest: *mut c_void,
    src: *const c_void,
    len: i32,
    dtype: MPI_Datatype,
) -> *mut c_void {
    debug_assert!(Context::is_init());
    debug_assert!(!dest.is_null() && !src.is_null());
    debug_assert!(len >= 0);
    MPI_CHECK_TYPE!(dtype, MPI_COMM_WORLD);

    unsafe {
        dest.copy_from(src, (len * p_mpi_type_size(dtype)) as usize);
        dest
    }
}

fn p_mpi_type_copy2(
    dest: *mut c_void,
    dcnt: i32,
    ddtype: MPI_Datatype,
    src: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
) -> *mut c_void {
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
pub extern "C" fn MPI_Barrier(comm: MPI_Comm) -> i32 {
    let mut size: i32 = uninit();
    let mut rank: i32 = uninit();

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
    mut root: i32,
    comm: MPI_Comm,
) -> i32 {
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);
    MPI_CHECK!(!buf.is_null(), comm, MPI_ERR_BUFFER);
    MPI_CHECK!(cnt >= 0, comm, MPI_ERR_COUNT);
    MPI_CHECK_TYPE!(dtype, comm);

    if Context::size() == 1 || cnt == 0 {
        return MPI_SUCCESS;
    }

    debug!("Begin Broadcast");
    let cg = Context::comm();
    cg.inc_key(comm);

    let mut code: i32;
    let mut stat: MPI_Status = uninit();

    if Context::size() == 2 {
        if Context::rank() == root {
            code = MPI_Send(buf, cnt, dtype, (root + 1) % 2, BCAST_TAG, comm);
        } else {
            code = MPI_Recv(buf, cnt, dtype, root, BCAST_TAG, comm, &mut stat);
        }
    } else {
        let mut n = 4;
        let diff = (Context::size() + Context::rank() - root) % Context::size();

        while n <= Context::size() {
            n <<= 1;
        }

        n >>= 1;
        loop {
            debug!("######N={n}");
            if Context::rank() == root {
                if diff + n < Context::size() {
                    debug!("#######Send to {}", (Context::rank() + n) % Context::size());
                    code = MPI_Send(
                        buf,
                        cnt,
                        dtype,
                        (Context::rank() + n) % Context::size(),
                        BCAST_TAG,
                        comm,
                    );
                    if code != MPI_SUCCESS {
                        break;
                    }
                }
            } else if Context::rank() == (root + n) % Context::size() {
                debug!("##########Recv from {}", root);
                code = MPI_Recv(buf, cnt, dtype, root, BCAST_TAG, comm, &mut stat);
                if code != MPI_SUCCESS {
                    break;
                }
                root = Context::rank();
            } else if diff > n {
                root = (root + n) % Context::size();
            }

            n >>= 1;
            if n <= 0 {
                code = MPI_SUCCESS;
                break;
            }
        }
    }

    debug!("End broadcast");

    cg.dec_key(comm);
    code
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
    MPI_CHECK_TYPE!(dtype, comm);

    if cnt == 0 {
        return MPI_SUCCESS;
    }

    if Context::size() == 1 {
        if p_mpi_type_copy(rbuf, sbuf, cnt, dtype).is_null() {
            MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
            return MPI_ERR_INTERN;
        }
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

        debug!("&&&Reduce begin");

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
                return MPI_ERR_INTERN;
            }
        }

        if diff % 2 != 0 {
            debug!(
                "Send to {}",
                (Context::size() + Context::rank() - 1) % Context::size()
            );
            code = MPI_Send(
                sbuf,
                cnt,
                dtype,
                (Context::size() + Context::rank() - 1) % Context::size(),
                REDUCE_TAG,
                comm,
            );
        } else if diff < (Context::size() - 1) {
            debug!("Recv from {}", (Context::rank() + 1) % Context::size());
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
                debug!("Do reduce function {cnt}, {dtype}");
                FUNCTIONS[op as usize](sbuf, rbuf, cnt, dtype);
            }
        }

        if code == MPI_SUCCESS {
            let mut tbuf: *mut c_void = null_mut();

            if diff % 4 != 0 {
                if diff % 2 == 0 {
                    if diff < Context::size() - 1 {
                        debug!(
                            "Write to {}",
                            (Context::size() + Context::rank() - 2) % Context::size()
                        );
                        code = MPI_Send(
                            rbuf,
                            cnt,
                            dtype,
                            (Context::size() + Context::rank() - 2) % Context::size(),
                            REDUCE_TAG,
                            comm,
                        );
                    } else {
                        debug!(
                            "Write to {}",
                            (Context::size() + Context::rank() - 2) % Context::size()
                        );
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
                    return MPI_ERR_INTERN;
                }

                debug!("Recv from {}", (Context::rank() + 2) % Context::size());
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
                    debug!("Do function");
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
                                debug!(
                                    "Send to: {}",
                                    (Context::size() + Context::rank() - iold) % Context::size()
                                );
                                code = MPI_Send(
                                    rbuf,
                                    cnt,
                                    dtype,
                                    (Context::size() + Context::rank() - iold) % Context::size(),
                                    REDUCE_TAG,
                                    comm,
                                );
                            } else {
                                debug!(
                                    "Send to: {}",
                                    (Context::size() + Context::rank() - iold) % Context::size()
                                );
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
                        debug!("Recv from: {}", (Context::rank() + iold) % Context::size());
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

    debug!("@@@Reduce end");
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
    CHECK_RET!(MPI_Reduce(sbuf, rbuf, cnt, dtype, op, 0, comm));
    CHECK_RET!(MPI_Bcast(rbuf, cnt, dtype, 0, comm));
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
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(root >= 0 && root < Context::size(), comm, MPI_ERR_ROOT);
    MPI_CHECK!(!sbuf.is_null() && !rbuf.is_null(), comm, MPI_ERR_BUFFER);
    MPI_CHECK!(scnt >= 0 && rcnt >= 0, comm, MPI_ERR_COUNT);
    MPI_CHECK_TYPE!(sdtype, comm);
    MPI_CHECK_TYPE!(rdtype, comm);

    if scnt == 0 {
        return MPI_SUCCESS;
    }

    if Context::size() == 1 {
        if p_mpi_type_copy2(rbuf, rcnt, rdtype, sbuf, scnt, sdtype).is_null() {
            MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
            return MPI_ERR_INTERN;
        }
        return MPI_SUCCESS;
    }

    let cg = Context::comm();
    cg.inc_key(comm);

    let mut stat: MPI_Status = uninit();
    let mut code;

    if Context::size() == 2 {
        if Context::rank() == root {
            let mut rext: i32 = uninit();
            code = MPI_Type_size(rdtype, &mut rext);
            if code != MPI_SUCCESS {
                cg.dec_key(comm);
                return code;
            }

            rext *= rcnt;

            code = MPI_Recv(
                unsafe { rbuf.add((rext * ((root + 1) % 2)) as usize) },
                rcnt,
                rdtype,
                (root + 1) % 2,
                GATHER_TAG,
                comm,
                &mut stat,
            );
            if code == MPI_SUCCESS {
                if p_mpi_type_copy2(
                    unsafe { rbuf.add((rext * root) as usize) },
                    rcnt,
                    rdtype,
                    sbuf,
                    scnt,
                    sdtype,
                )
                .is_null()
                {
                    cg.dec_key(comm);
                    MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                    return MPI_ERR_INTERN;
                }
            }
        } else {
            code = MPI_Send(sbuf, scnt, sdtype, root, GATHER_TAG, comm);
        }
    } else {
        if Context::rank() == root {
            let mut rext: i32 = uninit();
            code = MPI_Type_size(rdtype, &mut rext);
            if code != MPI_SUCCESS {
                cg.dec_key(comm);
                return code;
            }

            rext *= rcnt;

            if p_mpi_type_copy2(
                unsafe { rbuf.add((rext * Context::rank()) as usize) },
                rcnt,
                rdtype,
                sbuf,
                scnt,
                sdtype,
            )
            .is_null()
            {
                cg.dec_key(comm);
                MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                return MPI_ERR_INTERN;
            }

            code = MPI_Send(
                rbuf,
                rcnt * Context::size(),
                rdtype,
                (Context::rank() + 1) % Context::size(),
                GATHER_TAG,
                comm,
            );
            if code == MPI_SUCCESS {
                code = MPI_Recv(
                    rbuf,
                    rcnt * Context::size(),
                    rdtype,
                    (Context::size() + Context::rank() - 1) % Context::size(),
                    GATHER_TAG,
                    comm,
                    &mut stat,
                );
            }
        } else {
            let mut sext: i32 = uninit();
            code = MPI_Type_size(sdtype, &mut sext);
            if code != MPI_SUCCESS {
                cg.dec_key(comm);
                return code;
            }

            sext *= scnt;

            let layout =
                unsafe { Layout::from_size_align_unchecked((sext * Context::size()) as usize, 1) };
            rbuf = unsafe { std::alloc::alloc(layout) } as *mut c_void;
            if rbuf.is_null() {
                cg.dec_key(comm);
                MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                return MPI_ERR_INTERN;
            }

            code = MPI_Recv(
                rbuf,
                scnt * Context::size(),
                sdtype,
                (Context::size() + Context::rank() - 1) % Context::size(),
                GATHER_TAG,
                comm,
                &mut stat,
            );
            if code == MPI_SUCCESS {
                if p_mpi_type_copy(
                    unsafe { rbuf.add((sext * Context::rank()) as usize) },
                    sbuf,
                    scnt,
                    sdtype,
                )
                .is_null()
                {
                    cg.dec_key(comm);
                    MPI_Comm_call_errhandler(comm, MPI_ERR_INTERN);
                    return MPI_ERR_INTERN;
                }
                code = MPI_Send(
                    rbuf,
                    scnt * Context::size(),
                    sdtype,
                    (Context::rank() + 1) % Context::size(),
                    GATHER_TAG,
                    comm,
                );
            }
            unsafe {
                std::alloc::dealloc(rbuf as *mut u8, layout);
            }
        }
    }

    cg.dec_key(comm);
    code
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
    CHECK_RET!(MPI_Gather(sbuf, scnt, sdtype, rbuf, rcnt, rdtype, 0, comm));
    CHECK_RET!(MPI_Bcast(rbuf, rcnt * Context::size(), rdtype, 0, comm));
    MPI_SUCCESS
}
