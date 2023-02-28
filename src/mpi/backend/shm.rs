#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use libc::SYS_request_key;

use super::memory::memcpy;
use crate::{debug_bkd, debug_xfer, shared::*, xfer::request::Request};
use std::{mem::size_of, ptr::null_mut, sync::atomic::AtomicI8};

macro_rules! debug_shm {
    ($fmt:literal) => {
        debug_bkd!("shm", $fmt);
    };
    ($($args:tt)*) => {
        debug_bkd!("shm", $($args)*);
    }
}

#[repr(C)]
struct Cell {
    pub len: i32,                               // 4
    pub tag: i32,                               // 8
    pub m_flag: std::sync::atomic::AtomicUsize, // 16
    pub coll_flag: AtomicI8,                    // 17
    pub pad: [i8; 15],                          // 32
    pub buff: [i8; 8160],                       // 8192
}

impl Cell {
    pub const fn buf_len() -> usize {
        let m = null_mut() as *const Self;
        let p = unsafe { core::ptr::addr_of!((*m).buff) };

        const fn size_of_raw<T>(_: *const T) -> usize {
            core::mem::size_of::<T>()
        }
        size_of_raw(p)
    }

    pub fn set_coll_flag(&self, val: i8) {
        self.coll_flag
            .store(val, std::sync::atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn flag(&self) -> usize {
        self.m_flag.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[inline(always)]
    pub fn set_flag(&mut self, val: usize) {
        self.m_flag.store(val, std::sync::atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn dec_flag(&mut self) {
        self.m_flag
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn wait_eq(&self, target: usize) {
        while self.flag() != target {
            continue;
        }
    }

    #[inline(always)]
    pub fn wait_ne(&self, target: usize) {
        while self.flag() == target {
            continue;
        }
    }
}

#[repr(C)]
struct MpiShm {
    nsend: AtomicI8, // 1
    nrecv: AtomicI8, // 2
    pad: [i8; 30],   // 32
    cells: [Cell; 16],
}

impl MpiShm {
    #[inline(always)]
    pub fn swapSend(&mut self) {
        let idx = self.nsend.load(std::sync::atomic::Ordering::SeqCst);
        self.nsend.store(
            (idx + 1) % self.cells.len() as i8,
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    #[inline(always)]
    pub fn swapRecv(&mut self) {
        let idx = self.nrecv.load(std::sync::atomic::Ordering::SeqCst);
        self.nrecv.store(
            (idx + 1) % self.cells.len() as i8,
            std::sync::atomic::Ordering::SeqCst,
        );
    }

    #[inline(always)]
    pub fn coll_wait_and_swap(&mut self) {
        debug_shm!("Coll dec flag");
        let cell = self.recv_cell() as *mut Cell;
        let flag = unsafe { &mut (*cell).m_flag };
        let coll_flag = unsafe { &mut (*cell).coll_flag };
        debug_shm!(
            "Flag: {}, value: {}, coll_flag: {}",
            flag as *mut _ as usize,
            flag.load(std::sync::atomic::Ordering::SeqCst),
            coll_flag.load(std::sync::atomic::Ordering::SeqCst)
        );
        if coll_flag.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
            self.swapRecv();
            coll_flag.store(-1, std::sync::atomic::Ordering::SeqCst);
            while flag.load(std::sync::atomic::Ordering::SeqCst) != 1 {
                continue;
            }
            flag.store(0, std::sync::atomic::Ordering::SeqCst);
        } else {
            while coll_flag.load(std::sync::atomic::Ordering::SeqCst) != -1 {
                continue;
            }
            flag.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
        debug_shm!(
            "End coll_wait_and_swap, value: {}, coll_flag: {}",
            flag.load(std::sync::atomic::Ordering::SeqCst),
            coll_flag.load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    #[inline(always)]
    pub fn recv_cell(&mut self) -> &mut Cell {
        &mut self.cells[self.nrecv.load(std::sync::atomic::Ordering::SeqCst) as usize]
    }

    #[inline(always)]
    pub fn send_cell(&mut self) -> &mut Cell {
        &mut self.cells[self.nsend.load(std::sync::atomic::Ordering::SeqCst) as usize]
    }
}

pub struct ShmData {
    d: *mut MpiShm,
    shm_key: i32,
    recv_queue: RequestQueue,
    send_queue: RequestQueue,
    unexp_queue: RequestQueue,
}

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
            shm_key: -1,
            recv_queue: RequestQueue::new_c(),
            send_queue: RequestQueue::new_c(),
            unexp_queue: RequestQueue::new_c(),
        }
    }

    #[inline(always)]
    pub fn get_send(&mut self) -> Option<&mut Request> {
        self.send_queue.push()
    }

    #[inline(always)]
    pub fn get_recv(&mut self) -> Option<&mut Request> {
        self.recv_queue.push()
    }

    #[allow(unused_variables)]
    #[inline(always)]
    pub fn find_unexp(&mut self, rank: i32, tag: i32) -> Option<&mut Request> {
        if self.unexp_queue.len() != 0 {
            let val = unsafe { self.unexp_queue.iter().next().unwrap_unchecked() };
            debug_shm!(
                "Find unexpect: {rank}:{tag}, Data: {}:{}",
                val.rank,
                val.tag
            );
        }
        self.unexp_queue.find_by_tag(rank, tag)
    }

    pub fn free_req(&mut self, req: MPI_Request) {
        self.find_queue(req).erase_ptr(req);
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

    pub fn allocate_by_key(&mut self, key: i32) -> i32 {
        unsafe {
            let len = size_of::<MpiShm>() * (Context::size() * Context::size()) as usize;
            let mut id;
            if Context::rank() == 0 {
                id = libc::shmget(key, len, 0o666 | libc::IPC_CREAT);
                assert!(id != -1);
            } else {
                let errno = &*libc::__errno_location();
                loop {
                    id = libc::shmget(key, len, 0o666);
                    if id != -1 || *errno != libc::ENOENT {
                        break;
                    }
                }
                assert!(id != -1);
            }
            self.d = libc::shmat(id, null(), 0) as *mut MpiShm;
            if self.d as i64 == -1 {
                return !MPI_SUCCESS;
            }
            self.shm_key = id;
        }
        MPI_SUCCESS
    }

    pub fn deallocate(&mut self) -> MpiResult {
        unsafe {
            if self.shm_key == -1 {
                libc::munmap(
                    self.d as *mut c_void,
                    size_of::<MpiShm>() * (Context::size() * Context::size()) as usize,
                );
            } else {
                libc::shmdt(self.d as *mut c_void);
            }
        }
        Ok(())
    }

    pub fn init(&mut self, _: *mut i32, _: *mut *mut *mut i8, key: i32) -> MpiResult {
        debug_assert!(!Context::is_init());
        if key == -1 {
            if self.allocate() != MPI_SUCCESS {
                return Err(MPI_ERR_INTERN);
            }
        } else {
            if self.allocate_by_key(key) != MPI_SUCCESS {
                return Err(MPI_ERR_INTERN);
            }
        }
        Ok(())
    }

    pub fn deinit(&mut self) -> MpiResult {
        debug_assert!(Context::is_init());
        self.deallocate()?;
        Ok(())
    }

    pub fn progress(&mut self) -> MpiResult {
        let d = unsafe { &mut *(self as *mut Self) };

        for req in d.recv_queue.iter_mut() {
            Self::recv_progress(self as *mut Self, req)?;
        }

        for req in d.send_queue.iter_mut() {
            Self::send_progress(self as *mut Self, req)?;
        }

        Ok(())
    }

    #[inline(always)]
    fn recv_progress(this: *mut Self, mut req: &mut Request) -> MpiResult {
        if req.flag != 0 {
            return Ok(());
        }

        debug_shm!("Enter recover progress");

        let rank = Context::comm_rank(req.comm);
        let size = Context::comm_size(req.comm);

        let d = unsafe { &mut *this };
        let pshm = unsafe {
            if req.isColl {
                let rootRank = Context::comm_prank(req.comm, req.collRoot);
                d.d.add((rootRank * size + rootRank) as usize)
                    .as_mut()
                    .unwrap_unchecked()
            } else {
                d.d.add((req.rank * size + rank) as usize)
                    .as_mut()
                    .unwrap_unchecked()
            }
        };

        pshm.recv_cell().wait_ne(0);

        debug_shm!("Wait cell");

        let mut unexp = false;
        if req.tag != pshm.recv_cell().tag as i32 {
            debug_shm!(
                "Find unexpect message from rank: {}, {} != {}",
                req.rank,
                req.tag,
                pshm.recv_cell().tag
            );
            let preqx = d.unexp_queue.push();
            if let Some(reqx) = preqx {
                reqx.rank = req.rank;
                reqx.tag = pshm.recv_cell().tag as i32;
                reqx.cnt = pshm.recv_cell().len;

                req = reqx;

                unexp = true;
            } else {
                return Err(MPI_ERR_OTHER);
            }
        }

        if unexp {
            debug_shm!("Allocate unexpected buffer");
            let layout = std::alloc::Layout::from_size_align(req.cnt as usize, 32).unwrap();
            let buf = unsafe { std::alloc::alloc(layout) };
            if buf.is_null() {
                debug_shm!("Error allocate unexpected buffer");
                return Err(MPI_ERR_OTHER);
            }
            debug_assert!(
                buf as usize % 32 == 0,
                "Unexpected buff alignment: {}",
                buf as usize % 32
            );
            req.buf = buf as *mut c_void;
        } else if req.cnt < pshm.recv_cell().len {
            debug_shm!(
                "Truncate error for recv {} != {}",
                req.cnt,
                pshm.recv_cell().len
            );
            return Err(MPI_ERR_TRUNCATE);
        } else {
            req.cnt = pshm.recv_cell().len;
        }

        let mut length = req.cnt as usize;
        let mut buf = req.buf;

        debug_shm!("Recv length: {length}");
        while length > Cell::buf_len() {
            memcpy(
                buf,
                pshm.recv_cell().buff.as_ptr() as *const c_void,
                Cell::buf_len(),
            );
            if req.isColl {
                pshm.coll_wait_and_swap();
            } else {
                pshm.recv_cell().dec_flag();
                pshm.swapRecv();
            }
            pshm.recv_cell().wait_ne(0);

            buf = unsafe { buf.add(Cell::buf_len()) };
            length -= Cell::buf_len();
        }

        debug_assert!(pshm.recv_cell().buff.as_ptr() as *const c_void as usize % 32 == 0);
        memcpy(buf, pshm.recv_cell().buff.as_ptr() as *const c_void, length);
        if req.isColl {
            pshm.coll_wait_and_swap();
        } else {
            pshm.recv_cell().dec_flag();
            pshm.swapRecv();
        }

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        debug_shm!("Success recover from {}", req.tag);

        Ok(())
    }

    #[inline(always)]
    fn send_progress(this: *mut Self, req: &mut Request) -> MpiResult {
        if req.flag != 0 {
            return Ok(());
        }
        let rank = Context::comm_rank(req.comm);
        let size = Context::comm_size(req.comm);
        let flagValue: usize;

        let d = unsafe { &mut *this };
        let pshm = unsafe {
            if req.isColl {
                flagValue = size as usize - 1;
                let rootRank = Context::comm_prank(req.comm, req.collRoot);
                d.d.add((rootRank * size + rootRank) as usize)
                    .as_mut()
                    .unwrap_unchecked()
            } else {
                flagValue = 1;
                d.d.add((rank * size + req.rank) as usize)
                    .as_mut()
                    .unwrap_unchecked()
            }
        };

        pshm.send_cell().wait_eq(0);

        let mut length = req.cnt as usize;
        let mut buf = req.buf;
        pshm.send_cell().len = req.cnt;
        pshm.send_cell().tag = req.tag;
        debug_shm!("Send length: {length}");

        while length > Cell::buf_len() {
            memcpy(
                pshm.send_cell().buff.as_mut_ptr() as *mut c_void,
                buf,
                Cell::buf_len(),
            );
            if req.isColl {
                pshm.send_cell().set_coll_flag(flagValue as i8);
            }
            pshm.send_cell().set_flag(flagValue);
            pshm.swapSend();
            pshm.send_cell().wait_eq(0);

            buf = unsafe { buf.add(Cell::buf_len()) };
            length -= Cell::buf_len();
        }

        memcpy(
            pshm.send_cell().buff.as_mut_ptr() as *mut c_void,
            buf,
            length,
        );
        if req.isColl {
            pshm.send_cell().set_coll_flag(flagValue as i8);
        }
        pshm.send_cell().set_flag(flagValue);
        pshm.swapSend();

        req.stat.MPI_SOURCE = req.rank;
        req.stat.MPI_TAG = req.tag;
        req.stat.cnt = req.cnt;
        req.flag = 1;

        debug_shm!("Success send to {}", req.tag);

        Ok(())
    }
}
