use std::alloc::Layout;
use std::cmp::Ordering;
use std::mem::size_of;
use std::slice::from_raw_parts_mut;

use crate::context::Context;
use crate::{private::*, types::*, MPI_Allgather, MPI_Allreduce, MPI_CHECK, MPI_CHECK_COMM};

const KEY_INC: i32 = 2;

#[derive(Clone)]
struct Comm {
    prank: Vec<i32>,
    errh: MPI_Errhandler,
    rank: i32,
    key: i32,
}

impl Comm {
    const fn new() -> Comm {
        return Comm {
            prank: Vec::new(),
            errh: 0,
            rank: 0,
            key: 0,
        };
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
    col: i32,
    key: i32,
    rank: i32,
    grank: i32,
    key_max: i32,
}

impl CommSplit {
    const fn new() -> Self {
        uninit()
    }
}

pub struct CommGroup {
    comms: Vec<Comm>,
    key_max: i32,
}

impl CommGroup {
    pub const fn new() -> Self {
        CommGroup {
            comms: Vec::new(),
            key_max: 0,
        }
    }

    fn create_self(&mut self) -> i32 {
        debug_assert!(!Context::is_init());

        let comm = &mut self.comms[MPI_COMM_SELF as usize];

        comm.rank = 0;
        comm.key = self.key_max;
        comm.prank.push(Context::rank());

        self.key_max += KEY_INC;

        MPI_SUCCESS
    }

    fn create_world(&mut self) -> i32 {
        debug_assert!(!Context::is_init());

        let comm = &mut self.comms[MPI_COMM_WORLD as usize];

        comm.rank = Context::rank();
        comm.key = self.key_max;
        comm.prank.reserve(Context::size() as usize);

        for i in 0..Context::size() {
            comm.prank.push(i);
        }

        self.key_max += KEY_INC;

        MPI_SUCCESS
    }

    pub fn init(&mut self, _: *mut i32, _: *mut *mut *mut i8) -> i32 {
        debug_assert!(!Context::is_init());
        // debug_assert!(!pargc.is_null());
        // debug_assert!(!pargv.is_null());

        self.comms.resize(2, Comm::new());

        if self.create_self() == MPI_SUCCESS && self.create_world() == MPI_SUCCESS {
            return MPI_SUCCESS;
        }

        !MPI_SUCCESS
    }

    pub fn size(&self) -> usize {
        self.comms.len()
    }

    pub fn err_handler(&self, i: MPI_Comm) -> MPI_Errhandler {
        debug_assert!((i as usize) < self.size() && i >= 0);
        self.comms[i as usize].errh
    }

    pub fn set_err_handler(&mut self, comm: MPI_Comm, errh: MPI_Errhandler) {
        debug_assert!((comm as usize) < self.size() && comm >= 0);
        self.comms[comm as usize].errh = errh;
    }

    pub fn comm_dup(&mut self, comm: MPI_Comm, pcomm: *mut MPI_Comm) -> i32 {
        debug_assert!(Context::is_init());
        debug_assert!(comm >= 0 && comm < self.comms.len() as i32);
        debug_assert!(!pcomm.is_null());

        let mut key_max: i32 = uninit();

        CHECK_RET!(MPI_Allreduce(
            &self.key_max as *const i32 as *const c_void,
            &mut key_max as *mut i32 as *mut c_void,
            1,
            MPI_INT,
            MPI_MAX,
            comm
        ));

        self.comms.push(self.comms[comm as usize].clone());
        let item = unsafe { self.comms.last_mut().unwrap_unchecked() };
        item.key = key_max;

        unsafe { *pcomm = (self.comms.len() - 1) as i32 };
        self.key_max = key_max + KEY_INC;
        MPI_SUCCESS
    }

    fn split_cmp(lcomm: &CommSplit, rcomm: &CommSplit) -> Ordering {
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

    pub fn comm_split(&mut self, comm: MPI_Comm, col: i32, key: i32, pcomm: *mut MPI_Comm) -> i32 {
        debug_assert!(Context::is_init());
        debug_assert!(comm >= 0 && comm < self.comms.len() as i32);
        debug_assert!(col >= 0 || col == MPI_UNDEFINED);
        debug_assert!(!pcomm.is_null());

        let mut ent = CommSplit::new();
        let layout = unsafe {
            Layout::from_size_align_unchecked(
                self.comms[comm as usize].prank.len() * size_of::<CommSplit>(),
                size_of::<CommSplit>(),
            )
        };
        let pent = unsafe { std::alloc::alloc(layout) as *mut CommSplit };
        if pent.is_null() {
            return MPI_ERR_OTHER;
        }

        ent.col = col;
        ent.key = key;
        ent.rank = self.comms[comm as usize].rank;
        ent.grank = Context::rank();
        ent.key_max = self.key_max;

        let code = MPI_Allgather(
            &ent as *const CommSplit as *const c_void,
            5,
            MPI_INT,
            pent as *mut c_void,
            5,
            MPI_INT,
            comm,
        );
        if code != MPI_SUCCESS {
            unsafe { std::alloc::dealloc(pent as *mut u8, layout) };
            return code;
        }

        if col == MPI_UNDEFINED {
            unsafe {
                *pcomm = MPI_COMM_NULL;
                std::alloc::dealloc(pent as *mut u8, layout)
            };
            return MPI_SUCCESS;
        }

        let mut ncol = 0;
        let entarr = unsafe { from_raw_parts_mut(pent, self.comms[comm as usize].prank.len()) };
        for i in 0..self.comms[comm as usize].prank.len() {
            if entarr[i].col == col {
                ncol += 1;
            }
        }
        debug_assert!(ncol > 0);

        if ncol == 1 {
            self.comms.push(Comm::new());
            let pitem = unsafe { self.comms.last_mut().unwrap_unchecked() as *mut Comm };
            let item = unsafe { &mut *pitem };

            item.prank.push(Context::rank());
            item.rank = 0;
            item.key = self.key_max;
            item.errh = self.comms[comm as usize].errh;

            unsafe {
                *pcomm = (self.comms.len() - 1) as i32;
                self.key_max += KEY_INC;

                std::alloc::dealloc(pent as *mut u8, layout);
            }
            return MPI_SUCCESS;
        }

        let hlayout = unsafe {
            Layout::from_size_align_unchecked(
                ncol * size_of::<*mut CommSplit>(),
                size_of::<*mut CommSplit>(),
            )
        };
        let hent = unsafe { std::alloc::alloc(hlayout) as *mut *mut CommSplit };
        if hent.is_null() {
            unsafe { std::alloc::dealloc(pent as *mut u8, layout) };
            return MPI_ERR_OTHER;
        }

        let mut n = 0;
        let mut key_max = 0;
        let hentarr = unsafe { from_raw_parts_mut(hent, ncol) };
        for i in 0..self.comms[comm as usize].prank.len() {
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

        unsafe {
            hentarr.sort_unstable_by(|&a, &b| Self::split_cmp(&*a, &*b));
        }

        self.comms.push(Default::default());
        let pitem = unsafe { self.comms.last_mut().unwrap_unchecked() as *mut Comm };
        let item = unsafe { &mut *pitem };

        item.prank.reserve(ncol);
        for i in 0..ncol {
            unsafe {
                item.prank.push((*hentarr[i]).grank);
            }
        }
        for i in 0..ncol {
            if self.comms[comm as usize].rank == unsafe { (*hentarr[i]).rank } {
                item.rank = i as i32;
                break;
            }
        }

        item.key = self.key_max;
        item.errh = self.comms[comm as usize].errh;

        unsafe { *pcomm = (self.comms.len() - 1) as i32 };
        self.key_max = key_max + KEY_INC;

        unsafe {
            std::alloc::dealloc(hent as *mut u8, hlayout);
            std::alloc::dealloc(pent as *mut u8, layout);
        }
        MPI_SUCCESS
    }

    pub fn check(&self, comm: MPI_Comm) -> i32 {
        MPI_CHECK!(
            comm >= 0 && comm < self.comms.len() as i32,
            MPI_COMM_WORLD,
            MPI_ERR_COMM
        )
    }

    pub fn check_rank(&self, rank: i32, comm: MPI_Comm) -> i32 {
        let code = MPI_CHECK_COMM!(comm);
        if code != MPI_SUCCESS {
            return code;
        }
        MPI_CHECK!(
            rank >= 0 && rank < self.comms[comm as usize].prank.len() as i32,
            comm,
            MPI_ERR_RANK
        )
    }

    pub fn rank_map(&self, comm: MPI_Comm, rank: i32) -> i32 {
        debug_assert!(Context::is_init());
        self.check_rank(rank, comm);
        self.comms[comm as usize].prank[rank as usize]
    }

    pub fn rank_unmap(&self, comm: MPI_Comm, rank: i32) -> i32 {
        debug_assert!(Context::is_init());
        self.check_rank(rank, comm);
        for (i, r) in self.comms[comm as usize].prank.iter().enumerate() {
            if *r == rank {
                return i as i32;
            }
        }
        unreachable!()
    }

    pub fn tag_map(&self, comm: MPI_Comm, tag: i32) -> i32 {
        debug_assert!(Context::is_init());
        self.check(comm);
        debug_assert!(tag >= 0 && tag <= 32767);

        (self.comms[comm as usize].key << 16) | (tag & 0x7FFF)
    }

    pub fn tag_unmap(&self, comm: MPI_Comm, tag: i32) -> i32 {
        debug_assert!(Context::is_init());
        self.check(comm);

        tag & 0x7FFF
    }

    pub fn inc_key(&mut self, comm: MPI_Comm) {
        debug_assert!(Context::is_init());
        self.check(comm);

        self.comms[comm as usize].key += 1
    }

    pub fn dec_key(&mut self, comm: MPI_Comm) {
        debug_assert!(Context::is_init());
        self.check(comm);

        self.comms[comm as usize].key -= 1
    }
}

#[no_mangle]
pub extern "C" fn MPI_Comm_size(comm: MPI_Comm, psize: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!psize.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        psize.write(Context::size());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_rank(comm: MPI_Comm, prank: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!prank.is_null(), comm, MPI_ERR_ARG);

    unsafe {
        prank.write(Context::rank());
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_dup(comm: MPI_Comm, pcomm: *mut MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = Context::comm().comm_dup(comm, pcomm);
    if code != MPI_SUCCESS {
        Context::err_handler().call(comm, code);
    }

    code
}

#[no_mangle]
pub extern "C" fn MPI_Comm_split(comm: MPI_Comm, col: i32, key: i32, pcomm: *mut MPI_Comm) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(col >= 0 || col == MPI_UNDEFINED, comm, MPI_ERR_ARG);
    MPI_CHECK!(!pcomm.is_null(), comm, MPI_ERR_ARG);

    let code = Context::comm().comm_split(comm, col, key, pcomm);
    if code != MPI_SUCCESS {
        Context::err_handler().call(comm, code);
    }

    code
}

#[no_mangle]
pub extern "C" fn MPI_Comm_get_errhandler(comm: MPI_Comm, perrh: *mut MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    MPI_CHECK!(!perrh.is_null(), comm, MPI_ERR_ARG);

    unsafe { perrh.write(Context::comm().err_handler(comm)) }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Comm_set_errhandler(comm: MPI_Comm, errh: MPI_Errhandler) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_COMM!(comm);
    //  MPI_CHECK_ERRH!(comm, errh);

    Context::comm().set_err_handler(comm, errh);

    MPI_SUCCESS
}
