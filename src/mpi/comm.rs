use std::alloc::Layout;
use std::cmp::Ordering;
use std::mem::size_of;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::{types::*, private::*, MPI_CHECK, MPI_CHECK_COMM, MPI_CHECK_ERRH, MPI_Allreduce, uninit, MPI_Allgather};
use crate::context::Context;
use crate::errhandle::*;

const KEY_INC : i32 = 2;

#[derive(Clone)]
struct Comm {
    prank : Vec<i32>,
    errh : MPI_Errhandler,
    rank : i32,
    key : i32,
}

impl Comm {
    const fn new() -> Comm {
        return Comm{prank: Vec::new(), errh: 0, rank: 0, key: 0};
    }
}

impl Default for Comm {
    fn default() -> Self {
        return Self::new();
    }
}

unsafe impl Sync for Comm {}

#[derive(Clone, Copy)]
struct CommSplit {
    col : i32,
    key : i32,
    rank : i32,
    grank : i32,
    key_max : i32
}

impl CommSplit {
    const fn new() -> Self {
        uninit!()
    }
}

pub struct CommGroup {
    comms : Vec<Comm>,
    key_max : i32
}

static mut GROUP: CommGroup  = CommGroup{comms: Vec::new(), key_max: 0};

impl CommGroup {
    fn create_self() -> i32 {
        debug_assert!(!Context::is_init());

        unsafe {
            let comm = &mut GROUP.comms[MPI_COMM_SELF as usize];

            comm.rank = 0;
            comm.key = GROUP.key_max;
            comm.prank.push(Context::rank());

            GROUP.key_max += KEY_INC;
        }

        MPI_SUCCESS
    }

    fn create_world() -> i32 {
        debug_assert!(!Context::is_init());

        unsafe {
            let comm = &mut GROUP.comms[MPI_COMM_WORLD as usize];
            
            comm.rank = Context::rank();
            comm.key = GROUP.key_max;
            comm.prank.reserve(Context::size() as usize);

            for i in 0..Context::size() {
                comm.prank.push(i);
            }

            GROUP.key_max += KEY_INC;
        }

        MPI_SUCCESS
    }

    fn init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
        debug_assert!(!Context::is_init());
        debug_assert!(!pargc.is_null());
        debug_assert!(!pargv.is_null());

        unsafe {
            GROUP.comms.resize(2, Comm::new());
        }

        if Self::create_self() == MPI_SUCCESS && Self::create_world() == MPI_SUCCESS {
            return MPI_SUCCESS;
        }

        !MPI_SUCCESS
    }

    #[inline(always)]
    fn size() -> usize {
        unsafe {GROUP.comms.len()}
    }

    #[inline(always)]
    pub fn err_handler(i : MPI_Comm) -> MPI_Errhandler {
        debug_assert!((i as usize) < Self::size() && i >= 0);
        unsafe {GROUP.comms[i as usize].errh}
    }

    #[inline(always)]
    pub fn set_err_handler(comm : MPI_Comm, errh : MPI_Errhandler) {
        debug_assert!((comm as usize) < Self::size() && comm >= 0);
        unsafe {GROUP.comms[comm as usize].errh = errh};
    }

    pub fn comm_dup(comm : MPI_Comm, pcomm : *mut MPI_Comm) -> i32 {
        debug_assert!(Context::is_init());
        unsafe {
            debug_assert!(comm >= 0 && comm < GROUP.comms.len() as i32);
            debug_assert!(!pcomm.is_null());

            let mut key_max : i32 = uninit!();

            CHECK_RET!(MPI_Allreduce(&GROUP.key_max as *const i32 as *const c_void, &mut key_max as *mut i32 as *mut c_void, 1, MPI_INT, MPI_MAX, comm));

            GROUP.comms.push(GROUP.comms[comm as usize].clone());
            let item = GROUP.comms.last_mut().unwrap_unchecked();
            item.key = key_max;

            *pcomm = (GROUP.comms.len() - 1) as i32;
            GROUP.key_max = key_max + KEY_INC;
        }
        MPI_SUCCESS
    }

    fn split_cmp(lcomm : &CommSplit, rcomm : &CommSplit) -> Ordering {
        let keydiff = lcomm.key - rcomm.key;

        if keydiff < 0 {
            return Ordering::Less;
        } else if keydiff > 0 {
            return Ordering::Greater;
        }

        let rankdiff = lcomm.rank - rcomm.rank;
        if rankdiff < 0 {
            return Ordering::Less;
        } else if rankdiff > 0 {
            return Ordering::Greater;
        }
        Ordering::Equal
    }

    pub fn comm_split(comm : MPI_Comm, col : i32, key : i32, pcomm : *mut MPI_Comm) -> i32 {
        debug_assert!(Context::is_init());
        unsafe {
            debug_assert!(comm >= 0 && comm < GROUP.comms.len() as i32);
            debug_assert!(col >= 0 || col == MPI_UNDEFINED);
            debug_assert!(!pcomm.is_null());

            let mut ent = CommSplit::new();
            let layout = Layout::from_size_align_unchecked(GROUP.comms[comm as usize].prank.len() * size_of::<CommSplit>(), size_of::<CommSplit>());
            let pent = std::alloc::alloc(layout) as *mut CommSplit;
            if pent.is_null() {
                return MPI_ERR_OTHER;
            }

            ent.col = col;
            ent.key = key;
            ent.rank = GROUP.comms[comm as usize].rank;
            ent.grank = Context::rank();
            ent.key_max = GROUP.key_max;

            let code = MPI_Allgather(&ent as *const CommSplit as *const c_void, 5, MPI_INT, pent as *mut c_void, 5, MPI_INT, comm);
            if code != MPI_SUCCESS {
                std::alloc::dealloc(pent as *mut u8, layout);
                return code;
            }

            if col == MPI_UNDEFINED {
                *pcomm = MPI_COMM_NULL;
                std::alloc::dealloc(pent as *mut u8, layout);
                return MPI_SUCCESS
            }

            let mut ncol = 0;
            let entarr = from_raw_parts_mut(pent, GROUP.comms[comm as usize].prank.len());
            for i in 0..GROUP.comms[comm as usize].prank.len() {
                if entarr[i].col == col {
                    ncol += 1;
                }
            }
            debug_assert!(ncol > 0);

            if ncol == 1 {
                GROUP.comms.push(Comm::new());
                let item = GROUP.comms.last_mut().unwrap_unchecked();

                item.prank.push(Context::rank());
                item.rank = 0;
                item.key = GROUP.key_max;
                item.errh = GROUP.comms[comm as usize].errh;
                
                *pcomm = (GROUP.comms.len() - 1) as i32;
                GROUP.key_max += KEY_INC;

                std::alloc::dealloc(pent as *mut u8, layout);

                return MPI_SUCCESS;
            }

            let hlayout = Layout::from_size_align_unchecked(ncol * size_of::<*mut CommSplit>(), size_of::<*mut CommSplit>());
            let hent = std::alloc::alloc(hlayout) as *mut*mut CommSplit;
            if hent.is_null() {
                std::alloc::dealloc(pent as *mut u8, layout);
                return MPI_ERR_OTHER;
            }

            let mut n = 0;
            let mut key_max = 0;
            let hentarr = from_raw_parts_mut(hent, ncol);
            for i in 0..GROUP.comms[comm as usize].prank.len() {
                if entarr[i].col == col {
                    if key_max < entarr[i].key_max {
                        key_max = entarr[i].key_max;
                    }

                    hentarr[n] = &mut entarr[i];
                    n += 1;
                    if n == ncol {
                        break;
                    }
                }
            }

            hentarr.sort_unstable_by(|&a, &b| Self::split_cmp(&*a, &*b));

            GROUP.comms.push(Default::default());
            let item = GROUP.comms.last_mut().unwrap_unchecked();

            item.prank.reserve(ncol);
            for i in 0..ncol {
                item.prank.push((*hentarr[i]).grank);
            }
            for i in 0..ncol {
                if GROUP.comms[comm as usize].rank == (*hentarr[i]).rank {
                    item.rank = i as i32;
                    break;
                }
            }

            item.key = GROUP.key_max;
            item.errh = GROUP.comms[comm as usize].errh;

            *pcomm = (GROUP.comms.len() - 1) as i32;
            GROUP.key_max = key_max + KEY_INC;

            std::alloc::dealloc(hent as *mut u8, hlayout);
            std::alloc::dealloc(pent as *mut u8, layout);
        }
        MPI_SUCCESS
    }
}

pub (crate) fn p_mpi_check_comm(comm : MPI_Comm) -> i32
{
    unsafe {
        MPI_CHECK!(comm >= 0 && comm < GROUP.comms.len() as i32, MPI_COMM_WORLD, MPI_ERR_COMM)
    }
}

pub (crate) fn p_mpi_check_rank(rank : i32, comm : MPI_Comm) -> i32
{
    let code = MPI_CHECK_COMM!(comm);
    if code != MPI_SUCCESS {
        return code;
    }

    unsafe {
        MPI_CHECK!(rank >= 0 && rank < GROUP.comms[comm as usize].prank.len() as i32, comm, MPI_ERR_RANK)
    }
}

pub (crate) fn p_mpi_rank_map(comm : MPI_Comm, rank : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);
    unsafe {
        debug_assert!(rank >= 0 && rank < GROUP.comms[comm as usize].prank.len() as i32);

        GROUP.comms[comm as usize].prank[rank as usize]
    }
}

pub (crate) fn p_mpi_rank_unmap(comm : MPI_Comm, rank : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);
    debug_assert!(rank >= 0 && rank < Context::size());

    unsafe {
        for (i, r) in GROUP.comms[comm as usize].prank.iter().enumerate() {
            if *r == rank {
                return i as i32;
            }
        }
    }

    debug!("Unreacheble");

    unreachable!()
}

pub (crate) fn p_mpi_tag_map(comm : MPI_Comm, tag : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);
    debug_assert!(tag >= 0 && tag <= 32767);

    unsafe {
        (GROUP.comms[comm as usize].key << 16) | (tag & 0x7FFF)
    }
}

pub (crate) fn p_mpi_tag_unmap(comm : MPI_Comm, tag : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);

    tag & 0x7FFF
}

#[inline(always)]
pub (crate) fn p_mpi_comm_init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
    CommGroup::init(pargc, pargv)
}

pub (crate) fn p_mpi_comm_finit() -> i32 {
    debug_assert!(Context::is_init());

    MPI_SUCCESS
}

#[inline]
pub (crate) fn p_mpi_inc_key(comm : MPI_Comm)
{
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);

    unsafe {
        GROUP.comms[comm as usize].key += 1
    }
}

#[inline]
pub (crate) fn p_mpi_dec_key(comm : MPI_Comm)
{
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);

    unsafe {
        GROUP.comms[comm as usize].key -= 1
    }
}

#[no_mangle]
pub extern "C" fn MPI_Comm_size(comm : MPI_Comm, psize : *mut i32) -> i32
{
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!psize.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        psize.write(Context::size());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_rank(comm : MPI_Comm, prank : *mut i32) -> i32
{
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!prank.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        prank.write(Context::rank());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_dup(comm : MPI_Comm, pcomm : *mut MPI_Comm) -> i32
{
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = CommGroup::comm_dup(comm, pcomm);
    if code != MPI_SUCCESS {
        p_mpi_call_errhandler(comm, code);
    }

    code
}

#[no_mangle]
pub extern "C" fn MPI_Comm_split(comm : MPI_Comm, col : i32, key : i32, pcomm : *mut MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(col >= 0 || col == MPI_UNDEFINED, comm, MPI_ERR_ARG);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = CommGroup::comm_split(comm, col, key, pcomm);
    if code != MPI_SUCCESS {
        p_mpi_call_errhandler(comm, code);
    }

    code
}

#[no_mangle]
pub extern "C" fn MPI_Comm_get_errhandler(comm : MPI_Comm, perrh : *mut MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!perrh.is_null(), comm, MPI_ERR_ARG);

    unsafe { perrh.write(CommGroup::err_handler(comm)) }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_set_errhandler(comm : MPI_Comm, errh : MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK_ERRH!(comm, errh);

    CommGroup::set_err_handler(comm, errh);

    MPI_SUCCESS
}