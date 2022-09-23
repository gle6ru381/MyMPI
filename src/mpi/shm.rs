#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::{ffi::CStr, ptr::{read_volatile, write_volatile}};
use libc::c_void;

use crate::types::*;

const BufLen : usize = 4096;

struct ShmCell {
    pub len : i32,
    pub tag : i16,
    m_flag : i8,
    pub pad : i8,
    pub buff : [i8; 4096],
}

impl ShmCell {
    pub fn flag(& self) -> i8 {
        return unsafe {read_volatile::<i8>(&self.m_flag)};
    }
    pub fn setFlag(&mut self, val : i8) {
        return unsafe {write_volatile(&mut self.m_flag, val)};
    }
}

struct MpiShm {
    pub nsend : i8,
    pub nrecv : i8,
    pub cells : [ShmCell; 2],
}

impl MpiShm {
    pub fn swapSend(&mut self) {
        self.nsend = if self.nsend == 0 {1} else {0}
    }
    pub fn swapRecv(&mut self) {
        self.nrecv = if self.nrecv == 0 {1} else {0}
    }
}

pub struct ShmData {d : *mut MpiShm}

unsafe impl Sync for ShmData {}

pub struct MpiContext {
    pub shm : ShmData,
    pub mpiSize : i32,
    pub mpiRank : i32,
    pub MpiInit : bool
}

static mut context : MpiContext = MpiContext{shm: ShmData{d: std::ptr::null_mut()}, mpiSize: 1, mpiRank: 0, MpiInit : false};

impl MpiContext {

    pub fn parseArgs(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32
    {
        unsafe {
            let mut n : i32 = context.mpiSize;
            let mut argc = *pargc;
            let mut argv = *pargv;
        
            while --argc != 0 {
                argv = argv.add(1);
                let mut arg = CStr::from_ptr(*argv).to_str().unwrap();
                if arg == "-n" || arg == "-np" {
                    if (argc > 1) {
                        argv = argv.add(1);
                        arg = CStr::from_ptr(*argv).to_str().unwrap();
                        n = arg.parse::<i32>().unwrap();

                        let err = *libc::__errno_location();
                        match err {
                            libc::EINVAL | libc::ERANGE => n = context.mpiSize,
                            _ => {
                                if n > 0 {
                                    *pargc -= 2;
                                    while false {
                                        argv = argv.add(1);
                                        *((*pargv).add((*pargc - argc) as usize)) = *argv;
                                        *argv = std::ptr::null_mut();
                                    }

                                    println!("{}/{}: n = {n}", context.mpiRank, context.mpiSize);
                                } else {
                                    n = context.mpiSize;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            println!("Set mpi size on: {n}");
            context.mpiSize = n;
        }
        println!("Exit parse args");
        return MPI_SUCCESS;
    }

    pub fn allocate() -> i32 {
        unsafe {
            context.shm.d = libc::mmap(std::ptr::null_mut(), std::mem::size_of::<MpiShm>() * (context.mpiSize * context.mpiSize) as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANONYMOUS | libc::MAP_SHARED, -1, 0) as *mut MpiShm;
            return if context.shm.d as *mut c_void != libc::MAP_FAILED {MPI_SUCCESS} else {!MPI_SUCCESS};
        }
    }

    pub fn mpi_split(rank : i32, size : i32) -> i32
    {
        unsafe {context.mpiRank = rank};

        if size > 1 {
            unsafe {
                match libc::fork() {
                -1 => return !MPI_SUCCESS,
                0 => return Self::mpi_split(rank + size / 2 + size % 2, size / 2),
                _ => return Self::mpi_split(rank, size / 2 + size % 2)
                }
            }
        }

        unsafe {
            println!("{}/{}: mpi rank = {}", context.mpiRank, context.mpiSize, context.mpiRank);
        }

        return MPI_SUCCESS;
    }

    pub fn is_init() -> bool {
        return unsafe { context.MpiInit };
    }

    pub fn init() {
        unsafe {
            context.MpiInit = true;
        }
    }

    pub fn deinit() {
        unsafe {
            context.MpiInit = false;
        }
    }

    pub fn size() -> i32 {
        return unsafe {context.mpiSize};
    }

    pub fn rank() -> i32 {
        return unsafe {context.mpiRank};
    }

    pub fn send(buf : *const c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm) -> i32 {
        unsafe {
            assert!(dest >= 0 && dest < Self::size() && dest != Self::rank());
            assert!(dtype == MPI_BYTE);
            assert!(tag >= 0 && tag <= 32767);
            assert!(comm == MPI_COMM_WORLD);

            let mut pshm = context.shm.d.add((Self::rank() * Self::size() + dest) as usize).as_mut().unwrap();
            let mut cell = &mut pshm.cells[pshm.nsend as usize];

            while cell.flag() != 0
            {continue};

            cell.tag = tag as i16;
            cell.len = cnt;
            if cell.len  == 0 {
                cell.setFlag(1);
                pshm.swapSend();
            } else {
                let mut pbuf = buf;
                let mut len = cnt;
                while len > 0 {
                    while cell.flag() != 0 {
                        continue;
                    }

                    let ptr = (&mut cell.buff).as_mut_ptr();
                    let bytes = if (len as usize) < BufLen {cnt as usize} else {BufLen};

                    ptr.copy_from(pbuf as *const i8, bytes);
                    cell.setFlag(1);
                    pshm.swapSend();

                    pbuf = pbuf.add(BufLen);
                    len -= BufLen as i32;
                    cell = &mut pshm.cells[pshm.nsend as usize];
                }
                pshm.nsend = 0;
            }
        }

        MPI_SUCCESS
    }

    pub fn recv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, pstat : *mut MPI_Status) -> i32 {
        unsafe {
            let mut pshm = context.shm.d.add((src * Self::size() + Self::rank()) as usize).as_mut().unwrap();
            let mut cell = &mut pshm.cells[pshm.nrecv as usize];

            while cell.flag() == 0 {continue};

            if tag != cell.tag as i32 {
                println!("{}/{}: -> recv fail, tag {tag} != {}", Self::rank(), Self::size(), cell.tag);
                return !MPI_SUCCESS;
            }

            if cnt < cell.len {
                println!("{}/{}: -> recv fail, length {cnt} < {}", Self::rank(), Self::size(), cell.len);
                return !MPI_SUCCESS;
            }

            let mut length = cell.len;
            if length == 0 {
                cell.setFlag(0);
                pshm.swapRecv();
            } else {
                let mut pbuf = buf;
                while length > 0 {
                    while cell.flag() == 0 {continue};

                    let ptr = (& cell.buff).as_ptr();
                    let bytes = if (cnt as usize) < BufLen {cnt as usize} else {BufLen};
                    ptr.copy_to(pbuf as *mut i8, bytes);
                    cell.setFlag(0);
                    pshm.swapRecv();

                    pbuf = pbuf.add(BufLen);
                    length -= BufLen as i32;
                    cell = &mut pshm.cells[pshm.nrecv as usize];
                }

                pshm.nrecv = 0;
            }

            (*pstat).MPI_SOURCE = src;
            (*pstat).MPI_TAG = tag;
        }

        MPI_SUCCESS
    }
}