use crate::{types::*, MPI_CHECK, MPI_CHECK_COMM, MPI_CHECK_ERRH};
use crate::context::Context;
use crate::errhandle::*;

const CTXT_INC : i32 = 2;
const COMM_MAX : usize = 3;

#[derive(Clone)]
struct Comm {
    prank : Vec<i32>,
    errh : MPI_Errhandler,
    rank : i32,
    ctxt : i32,
}

impl Comm {
    const fn new() -> Comm {
        return Comm{prank: Vec::new(), errh: 0, rank: 0, ctxt: 0};
    }
}

impl Default for Comm {
    fn default() -> Self {
        return Self::new();
    }
}

unsafe impl Sync for Comm {}

pub struct CommGroup {
    comms : Vec<Comm>,
    ctxt_max : i32
}

static mut GROUP: CommGroup  = CommGroup{comms: Vec::new(), ctxt_max: 0};

impl CommGroup {
    fn create_self() -> i32 {
        debug_assert!(!Context::is_init());

        unsafe {
            let comm = &mut GROUP.comms[MPI_COMM_SELF as usize];

            comm.rank = 0;
            comm.ctxt = GROUP.ctxt_max;
            comm.prank.push(Context::rank());

            GROUP.ctxt_max += CTXT_INC;
        }

        MPI_SUCCESS
    }

    fn create_world() -> i32 {
        debug_assert!(!Context::is_init());

        unsafe {
            let comm = &mut GROUP.comms[MPI_COMM_WORLD as usize];
            
            comm.rank = Context::rank();
            comm.ctxt = GROUP.ctxt_max;
            comm.prank.reserve(Context::size() as usize);

            for i in 0..Context::size() {
                comm.prank.push(i);
            }

            GROUP.ctxt_max += CTXT_INC;
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

pub (crate) fn p_mpi_get_grank(comm : MPI_Comm, lrank : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);
    unsafe {
        debug_assert!(lrank >= 0 && lrank < GROUP.comms[comm as usize].prank.len() as i32);

        GROUP.comms[comm as usize].prank[lrank as usize]
    }
}

pub (crate) fn p_mpi_get_gtag(comm : MPI_Comm, ltag : i32) -> i32 {
    debug_assert!(Context::is_init());
    MPI_CHECK_COMM!(comm);
    debug_assert!(ltag >= 0 && ltag <= 32767);

    unsafe {
        (GROUP.comms[comm as usize].ctxt << 16) | (ltag & 0x7FFF)
    }
}

#[inline(always)]
pub (crate) fn p_mpi_comm_init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
    CommGroup::init(pargc, pargv)
}

pub (crate) fn p_mpi_comm_finit() -> i32 {
    debug_assert!(Context::is_init());

    MPI_SUCCESS
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
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_split(comm : MPI_Comm, col : i32, key : i32, pcomm : *mut MPI_Comm) -> i32 {
    MPI_SUCCESS
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