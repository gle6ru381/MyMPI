#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::{ptr::{read_volatile, write_volatile, null, null_mut}, mem::size_of, collections::HashSet};
use libc::c_void;

use crate::{types::*, context::Context};

macro_rules! field_size {
    ($t:ident :: $field:ident) => {{
        let m = core::mem::MaybeUninit::<$t>::uninit();
        let p = unsafe {
            core::ptr::addr_of!((*(&m as *const _ as *const $t)).$field)
        };

        const fn size_of_raw<T>(_: *const T) -> usize {
            core::mem::size_of::<T>()
        }
        size_of_raw(p)
    }};
}

struct Cell {
    pub len : i32,
    pub tag : i16,
    m_flag : i8,
    pub pad : i8,
    pub buff : [i8; 4096],
}

impl Cell {
    pub const fn buf_len() -> usize {
        field_size!(Cell::buff)
    } 

    #[inline(always)]
    pub fn flag(& self) -> i8 {
        return unsafe {read_volatile(&self.m_flag)};
    }

    #[inline(always)]
    pub fn setFlag(&mut self, val : i8) {
        return unsafe {write_volatile(&mut self.m_flag, val)};
    }

    #[inline(always)]
    pub fn wait_eq(&self, target : i8) {
        while self.flag() != target {continue;}
    }

    #[inline(always)]
    pub fn wait_ne(&self, target : i8) {
        while self.flag() == target {continue;}
    }
}

struct MpiShm {
    nsend : i8,
    nrecv : i8,
    cells : [Cell; 2],
}

impl MpiShm {
    #[inline(always)]
    fn p_swap(nsend : &mut i8, size : i8) {
        *nsend = (*nsend + 1) % size;
    }

    #[inline(always)]
    pub fn swapSend(&mut self) {
        Self::p_swap(&mut self.nsend, self.cells.len() as i8);
    }

    #[inline(always)]
    pub fn swapRecv(&mut self) {
        self.nrecv = (self.nrecv + 1) % self.cells.len() as i8;
    }

    #[inline(always)]
    pub fn recv_cell(&mut self) -> &mut Cell {
        &mut self.cells[self.nrecv as usize]
    }

    #[inline(always)]
    pub fn send_cell(&mut self) -> &mut Cell {
        &mut self.cells[self.nsend as usize]
    }
}

struct RequestQueue {
    reqs : [P_MPI_Request; 1],
    fill : i32
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue {
    pub const fn new () -> Self {
        RequestQueue { reqs: [P_MPI_Request::new(); 1], fill: 0 }
    }

    #[inline(always)]
    pub fn find_by_tag(&mut self, rank : i32, tag : i32) -> MPI_Request {
        let val = self.requests().iter().position(|x| x.rank == rank && x.tag == tag );
        return if val.is_none() {null_mut()} else {&mut self.reqs[val.unwrap()]}
    }

    #[inline(always)]
    pub fn find_mut(&mut self, req : MPI_Request) -> MPI_Request {
       let val = self.requests().iter().position(|x| x as *const P_MPI_Request == req);
       return if val.is_none() {null_mut()} else {&mut self.reqs[val.unwrap()]}
    }

    #[inline(always)]
    pub fn find(&self, req : MPI_Request) -> *const P_MPI_Request {
        let val = self.requests_const().iter().find(|&x| x as *const P_MPI_Request == req);
        return if val.is_none() {null_mut()} else {val.unwrap()}
    }

    #[inline(always)]
    pub fn contains(&self, req : MPI_Request) -> bool {
        return self.find(req) != null_mut();
    }

    #[inline(always)]
    pub fn get(&mut self) -> MPI_Request {
        if self.fill < 1 {
            let idx = self.fill as usize;
            self.fill += 1;
            return &mut self.reqs[idx];
        }
        return null_mut();
    }

    #[inline(always)]
    pub fn free(&mut self, req : MPI_Request) {
        self.fill = 0;
    }

    #[inline(always)]
    pub fn requests(&mut self) -> &mut [P_MPI_Request] {
        &mut self.reqs[..self.fill as usize]
    }
    
    #[inline(always)]
    pub fn requests_const(&self) -> &[P_MPI_Request] {
        &self.reqs[..self.fill as usize]
    }
}

pub struct ShmData {
    d : *mut MpiShm,
    recv_queue : RequestQueue,
    send_queue : RequestQueue,
    unexp_queue : RequestQueue
}

unsafe impl Sync for ShmData {}

impl ShmData {
    fn find_queue(&mut self, req : MPI_Request) -> &mut RequestQueue {
        if &self.recv_queue.reqs[0] as *const P_MPI_Request == req {
            return &mut self.recv_queue;
        }
        if &self.send_queue.reqs[0] as *const P_MPI_Request == req {
            return &mut self.send_queue;
        }
        if &self.unexp_queue.reqs[0] as *const P_MPI_Request == req {
            return &mut self.unexp_queue;
        }

        unreachable!();
    }

    pub const fn new() -> ShmData {
        ShmData { d: std::ptr::null_mut(), recv_queue: RequestQueue::new(), send_queue: RequestQueue::new(), unexp_queue: RequestQueue::new() }
    }

    pub fn get_send(&mut self) -> MPI_Request {
        self.send_queue.get()
    }

    pub fn get_recv(&mut self) -> MPI_Request {
        self.recv_queue.get()
    }

    pub fn find_unexp(&mut self, rank : i32, tag : i32) -> MPI_Request {
        self.unexp_queue.find_by_tag(rank, tag)
    }

    pub fn free_req(&mut self, req : MPI_Request,  preq : MPI_Request) {
        self.find_queue(req).free(preq);
    }

    pub fn allocate(&mut self) -> i32 {
        unsafe {
            self.d = libc::mmap(std::ptr::null_mut(), size_of::<MpiShm>() * (Context::size() * Context::size()) as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANONYMOUS | libc::MAP_SHARED, -1, 0) as *mut MpiShm;
            if self.d as *mut c_void == libc::MAP_FAILED {
                return !MPI_SUCCESS;
            }
        }
        MPI_SUCCESS
    }

    pub fn init(&mut self, pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
        unsafe {
            debug_assert!(!Context::is_init());
            if self.allocate() != MPI_SUCCESS {
                return MPI_ERR_INTERN;
            }
        }
        MPI_SUCCESS
    }

    pub fn progress(&mut self) -> i32 {
        println!("Call progress for {}, send: {}, recv: {}", Context::rank(), self.send_queue.requests().len(), self.recv_queue.requests().len());
        let d = unsafe {&mut *(self as *mut Self)};
        for req in d.recv_queue.requests() {
            let rc = Self::recv_progress(self as *mut Self, req);
            if rc != MPI_SUCCESS {
                return rc;
            }
        }

        for req in d.send_queue.requests() {
            let rc = Self::send_progress(self as *mut Self, req);
            if rc != MPI_SUCCESS {
                return rc;
            }
        }

        MPI_SUCCESS
    }

    #[inline(always)]
    fn recv_progress(this : *mut Self, req : &mut P_MPI_Request) -> i32 {
        if req.flag != 0 {
            return MPI_SUCCESS
        }

        println!("Shm recv progress");

        let d = unsafe {&mut *this};
        let pshm = unsafe {d.d.add((req.rank * Context::size() + Context::rank()) as usize).as_mut().unwrap()};

        pshm.recv_cell().wait_ne(0);

        let mut unexp = false;
        if req.tag != pshm.recv_cell().tag as i32 {
            println!("Find enexpect");
            let preqx = d.unexp_queue.get();
            if !preqx.is_null() {
                let reqx = unsafe {&mut *preqx};
                reqx.rank = req.rank;
                reqx.tag = pshm.recv_cell().tag as i32;
                reqx.cnt = pshm.recv_cell().len;

                unexp = true;
            } else {
                return MPI_ERR_OTHER;
            }
        }

        if unexp {
            let layout = std::alloc::Layout::from_size_align(req.cnt as usize, 1).unwrap();
            let buf = unsafe {std::alloc::alloc(layout)};
            if buf.is_null() {
                return MPI_ERR_OTHER;
            }
            req.buf = buf as *mut c_void;
        } else if req.cnt < pshm.recv_cell().len {
            println!("Truncate error for recv");
            return MPI_ERR_TRUNCATE;
        } else {
            req.cnt = pshm.recv_cell().len;
        }

        let mut length = req.cnt as usize;
        let mut buf = req.buf;
        println!("Recv length: {length}");
        while length > Cell::buf_len() {
            unsafe {
                buf.copy_from(pshm.recv_cell().buff.as_ptr() as *const c_void, Cell::buf_len());
            }
            pshm.recv_cell().setFlag(0);
            pshm.swapRecv();
            pshm.recv_cell().wait_ne(0);

            buf = unsafe {buf.add(Cell::buf_len())};
            length -= Cell::buf_len();
        }

        unsafe {
            buf.copy_from(pshm.recv_cell().buff.as_ptr() as *const c_void, length);
        }
        pshm.recv_cell().setFlag(0);
        pshm.swapRecv();

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        MPI_SUCCESS
    }

    #[inline(always)]
    fn send_progress(this : *mut Self, req : &mut P_MPI_Request) -> i32 {
        if req.flag != 0 {
            return MPI_SUCCESS
        }

        println!("Shm send progress");

        let d = unsafe {&mut *this};
        let pshm = unsafe {d.d.add((Context::rank() * Context::size() + req.rank) as usize).as_mut().unwrap_unchecked()};

        pshm.send_cell().wait_eq(0);

        let mut length = req.cnt as usize;
        let mut buf = req.buf;
        pshm.send_cell().len = req.cnt;
        println!("Send length: {length}");

        while length > Cell::buf_len() {
            unsafe {
                buf.copy_to(pshm.send_cell().buff.as_mut_ptr() as *mut c_void, Cell::buf_len());
            }
            pshm.send_cell().setFlag(1);
            pshm.swapSend();
            pshm.send_cell().wait_eq(0);

            buf = unsafe {buf.add(Cell::buf_len())};
            length -= Cell::buf_len();
        }

        unsafe {
            buf.copy_to(pshm.send_cell().buff.as_mut_ptr() as *mut c_void, length);
        }
        pshm.send_cell().setFlag(1);
        pshm.swapSend();

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        MPI_SUCCESS
    }
}