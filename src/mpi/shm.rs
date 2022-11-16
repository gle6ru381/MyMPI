#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use crate::memory::memcpy;
use crate::private::*;
use std::{
    mem::size_of,
    ptr::{null_mut, read_volatile, write_volatile},
};

macro_rules! field_size {
    ($t:ident :: $field:ident) => {{
        let m = core::mem::MaybeUninit::<$t>::uninit();
        let p = unsafe { core::ptr::addr_of!((*(&m as *const _ as *const $t)).$field) };

        const fn size_of_raw<T>(_: *const T) -> usize {
            core::mem::size_of::<T>()
        }
        size_of_raw(p)
    }};
}

#[repr(C)]
struct Cell {
    pub len: i32,             // 4
    pub tag: i32,             // 8
    m_flag: i8,               // 9
    pub pad: [i8; 23],        // 32
    pub buff: [i8; 16777184], // 16 777 216
}

impl Cell {
    pub const fn buf_len() -> usize {
        field_size!(Cell::buff)
    }

    #[inline(always)]
    pub fn flag(&self) -> i8 {
        return unsafe { read_volatile(&self.m_flag) };
    }

    #[inline(always)]
    pub fn setFlag(&mut self, val: i8) {
        return unsafe { write_volatile(&mut self.m_flag, val) };
    }

    #[inline(always)]
    pub fn wait_eq(&self, target: i8) {
        while self.flag() != target {
            continue;
        }
    }

    #[inline(always)]
    pub fn wait_ne(&self, target: i8) {
        while self.flag() == target {
            continue;
        }
    }
}

#[repr(C)]
struct MpiShm {
    nsend: i8,     // 1
    nrecv: i8,     // 2
    pad: [i8; 30], // 32
    cells: [Cell; 2],
}

impl MpiShm {
    #[inline(always)]
    pub fn swapSend(&mut self) {
        self.nsend = (self.nsend + 1) % self.cells.len() as i8;
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

pub struct ShmData {
    d: *mut MpiShm,
    recv_queue: RequestQueue,
    send_queue: RequestQueue,
    unexp_queue: RequestQueue,
}

unsafe impl Sync for ShmData {}

impl ShmData {
    fn find_queue(&mut self, req: MPI_Request) -> &mut RequestQueue {
        if self.recv_queue.contains(req) {
            return &mut self.recv_queue;
        }
        if self.send_queue.contains(req) {
            return &mut self.send_queue;
        }
        if self.unexp_queue.contains(req) {
            return &mut self.unexp_queue;
        }

        unreachable!();
    }

    pub const fn new() -> ShmData {
        ShmData {
            d: null_mut(),
            recv_queue: RequestQueue::new_c(),
            send_queue: RequestQueue::new_c(),
            unexp_queue: RequestQueue::new_c(),
        }
    }

    #[inline(always)]
    pub fn get_send(&mut self) -> Option<&mut P_MPI_Request> {
        self.send_queue.push()
    }

    #[inline(always)]
    pub fn get_recv(&mut self) -> Option<&mut P_MPI_Request> {
        self.recv_queue.push()
    }

    #[allow(unused_variables)]
    #[inline(always)]
    pub fn find_unexp(&mut self, rank: i32, tag: i32) -> MPI_Request {
        if self.unexp_queue.len() != 0 {
            let val = self.unexp_queue.iter().next().unwrap();
            debug!(
                "Find unexpect: {rank}:{tag}, Data: {}:{}",
                val.rank, val.tag
            );
        }
        self.unexp_queue.find_by_tag(rank, tag)
    }

    pub fn free_req(&mut self, req: MPI_Request, preq: MPI_Request) {
        self.find_queue(req).erase_ptr(preq);
    }

    pub fn allocate(&mut self) -> i32 {
        unsafe {
            self.d = libc::mmap(
                null_mut(),
                size_of::<MpiShm>() * (Context::size() * Context::size()) as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_SHARED,
                -1,
                0,
            ) as *mut MpiShm;
            if self.d as *mut c_void == libc::MAP_FAILED {
                return !MPI_SUCCESS;
            }
        }
        MPI_SUCCESS
    }

    pub fn deallocate(&mut self) -> i32 {
        unsafe {
            libc::munmap(
                self.d as *mut c_void,
                size_of::<MpiShm>() * (Context::size() * Context::size()) as usize,
            );
        }
        MPI_SUCCESS
    }

    pub fn init(&mut self, _: *mut i32, _: *mut *mut *mut i8) -> i32 {
        debug_assert!(!Context::is_init());
        if self.allocate() != MPI_SUCCESS {
            return MPI_ERR_INTERN;
        }
        MPI_SUCCESS
    }

    pub fn deinit(&mut self) -> i32 {
        debug_assert!(Context::is_init());
        if self.deallocate() != MPI_SUCCESS {
            return MPI_ERR_INTERN;
        }
        MPI_SUCCESS
    }

    pub fn progress(&mut self) -> i32 {
        let d = unsafe { &mut *(self as *mut Self) };

        for req in d.recv_queue.iter_mut() {
            let rc = Self::recv_progress(self as *mut Self, req);
            if rc != MPI_SUCCESS {
                return rc;
            }
        }

        for req in d.send_queue.iter_mut() {
            let rc = Self::send_progress(self as *mut Self, req);
            if rc != MPI_SUCCESS {
                return rc;
            }
        }

        MPI_SUCCESS
    }

    #[inline(always)]
    fn recv_progress(this: *mut Self, mut req: &mut P_MPI_Request) -> i32 {
        if req.flag != 0 {
            return MPI_SUCCESS;
        }

        let d = unsafe { &mut *this };
        let pshm = unsafe {
            d.d.add((req.rank * Context::size() + Context::rank()) as usize)
                .as_mut()
                .unwrap()
        };

        debug!("Recv flag: {}", pshm.recv_cell().flag());
        pshm.recv_cell().wait_ne(0);

        let mut unexp = false;
        if req.tag != pshm.recv_cell().tag as i32 {
            debug!("{} Find unexpect", Context::rank());
            let preqx = d.unexp_queue.push();
            if preqx.is_some() {
                let reqx = unsafe { preqx.unwrap_unchecked() };
                reqx.rank = req.rank;
                reqx.tag = pshm.recv_cell().tag as i32;
                reqx.cnt = pshm.recv_cell().len;

                req = reqx;

                unexp = true;
            } else {
                return MPI_ERR_OTHER;
            }
        }

        if unexp {
            let layout = std::alloc::Layout::from_size_align(req.cnt as usize, 32).unwrap();
            let buf = unsafe { std::alloc::alloc(layout) };
            if buf.is_null() {
                debug!("Error allocate unexpected buffer");
                return MPI_ERR_OTHER;
            }
            debug_assert!(
                buf as usize % 32 == 0,
                "Unexpected buff alignment: {}",
                buf as usize % 32
            );
            req.buf = buf as *mut c_void;
        } else if req.cnt < pshm.recv_cell().len {
            debug!("Truncate error for recv");
            return MPI_ERR_TRUNCATE;
        } else {
            req.cnt = pshm.recv_cell().len;
        }

        let mut length = req.cnt as usize;
        let mut buf = req.buf;
        debug_assert!(buf as usize % 32 == 0 || length == 0, "Length: {length}");
        debug!("Recv length: {length}");
        while length > Cell::buf_len() {
            memcpy(
                buf,
                pshm.recv_cell().buff.as_ptr() as *const c_void,
                Cell::buf_len(),
            );
            pshm.recv_cell().setFlag(0);
            pshm.swapRecv();
            pshm.recv_cell().wait_ne(0);

            buf = unsafe { buf.add(Cell::buf_len()) };
            length -= Cell::buf_len();
        }

        debug_assert!(
            buf as usize % 32 == 0 || length == 0,
            "Buff alignment: {}",
            buf as usize % 32
        );
        debug_assert!(pshm.recv_cell().buff.as_ptr() as *const c_void as usize % 32 == 0);
        memcpy(buf, pshm.recv_cell().buff.as_ptr() as *const c_void, length);
        pshm.recv_cell().setFlag(0);
        pshm.swapRecv();

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        debug!(
            "Success recover for {}, buffer: {}",
            Context::rank(),
            req.buf.is_null()
        );

        MPI_SUCCESS
    }

    #[inline(always)]
    fn send_progress(this: *mut Self, req: &mut P_MPI_Request) -> i32 {
        if req.flag != 0 {
            return MPI_SUCCESS;
        }

        let d = unsafe { &mut *this };
        let pshm = unsafe {
            d.d.add((Context::rank() * Context::size() + req.rank) as usize)
                .as_mut()
                .unwrap_unchecked()
        };
        debug!("Send cell flag: {}", pshm.send_cell().flag());
        pshm.send_cell().wait_eq(0);

        let mut length = req.cnt as usize;
        let mut buf = req.buf;
        pshm.send_cell().len = req.cnt;
        pshm.send_cell().tag = req.tag;
        debug!("Send length: {length}");

        while length > Cell::buf_len() {
            memcpy(
                pshm.send_cell().buff.as_mut_ptr() as *mut c_void,
                buf,
                Cell::buf_len(),
            );
            pshm.send_cell().setFlag(1);
            pshm.swapSend();
            pshm.send_cell().wait_eq(0);

            buf = unsafe { buf.add(Cell::buf_len()) };
            length -= Cell::buf_len();
        }

        debug_assert!(
            pshm.send_cell().buff.as_mut_ptr() as usize % 32 == 0,
            "Current: {}, Cell align: {}",
            pshm.send_cell().buff.as_mut_ptr() as usize % 32,
            pshm.send_cell() as *mut Cell as usize % 32
        );

        memcpy(
            pshm.send_cell().buff.as_mut_ptr() as *mut c_void,
            buf,
            length,
        );
        pshm.send_cell().setFlag(1);
        pshm.swapSend();

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        debug!("Success send for {}", Context::rank());

        MPI_SUCCESS
    }
}
