#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::{ffi::CStr};
use libc::c_void;

use crate::types::*;

const BufLen : usize = 4096;

struct MpiShm {
    pub len : i32,
    pub tag : i16,
    pub pad : i16,
    m_flags : [i8; 2],
    pub nsend : i8,
    pub nrecv : i8,
    pub buff : [[i8; 4096]; 2]
}

pub struct ShmData {d : *mut MpiShm}

unsafe impl Sync for ShmData {}

impl MpiShm {

    pub fn Default() -> MpiShm {
        let d = std::mem::MaybeUninit::<MpiShm>::uninit();
        unsafe {
            let mut shm = d.assume_init();
            shm.len = 0;
            shm.tag = 0;
            shm.pad = 0;
            shm.m_flags = [0, 0];
            shm.nsend = 0;
            shm.nrecv = 0;
            return shm;
        }
    }

    pub fn flags(&self, idx : usize) -> i8 {
        unsafe {
            let mut d = std::ptr::read_volatile(&self.m_flags[idx]);
            return d
        }
    }

    pub fn setFlag(&mut self, idx : usize,  val : i8) {
        unsafe {
            std::ptr::write_volatile(&mut self.m_flags[idx], val);
        }
    }
}

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

    pub fn send(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, dest : i32, tag : i32, comm : MPI_Comm) -> i32 {
        unsafe {
            let mut pshm = context.shm.d.add((Self::rank() * Self::size() + dest) as usize).as_mut().unwrap();

            while pshm.flags(pshm.nsend as usize) != 0
            {continue};

            pshm.tag = tag as i16;
            pshm.len = cnt;

            if pshm.len == 0 {
                pshm.setFlag(pshm.nsend as usize, 1);
            } else {
                let mut pbuf = buf;
                let mut len = cnt;
                while len > 0 {
                    while pshm.flags(pshm.nsend as usize) != 0 {
                        continue;
                    }

                    let ptr = (&mut pshm.buff[pshm.nsend as usize]).as_mut_ptr();
                    let bytes = if (len as usize) < BufLen {cnt as usize} else {BufLen};

                    ptr.copy_from(pbuf as *const i8, bytes);
                    pshm.setFlag(pshm.nsend as usize, 1);
                    pshm.nsend = !pshm.nsend;

                    pbuf = pbuf.add(BufLen);
                    len -= BufLen as i32;
                }
                pshm.nsend = 0;
            }
        }

        MPI_SUCCESS
    }

    pub fn recv(buf : *mut c_void, cnt : i32, dtype : MPI_Datatype, src : i32, tag : i32, comm : MPI_Comm, pstat : *mut MPI_Status) -> i32 {
        unsafe {
            let mut pshm = context.shm.d.add((src * Self::size() + Self::rank()) as usize).as_mut().unwrap();

            while pshm.flags(pshm.nrecv as usize) == 0 {continue};

            if tag != pshm.tag as i32 {
                println!("{}/{}: -> recv fail, tag {tag} != {}", Self::rank(), Self::size(), pshm.tag);
                return !MPI_SUCCESS;
            }

            if cnt < pshm.len {
                println!("{}/{}: -> recv fail, length {cnt} < {}", Self::rank(), Self::size(), pshm.len);
                return !MPI_SUCCESS;
            }

            let mut length = pshm.len;
            if length == 0 {
                pshm.setFlag(pshm.nrecv as usize, 0);
            } else {
                let mut pbuf = buf;
                while length > 0 {
                    while pshm.flags(pshm.nrecv as usize) == 0 {continue};

                    let ptr = (& pshm.buff[pshm.nrecv as usize]).as_ptr();
                    let bytes = if (cnt as usize) < BufLen {cnt as usize} else {BufLen};
                    ptr.copy_to(pbuf as *mut i8, bytes);
                    pshm.setFlag(pshm.nrecv as usize, 0);
                    pshm.nrecv = !pshm.nrecv;

                    pbuf = pbuf.add(BufLen);
                    length -= BufLen as i32;
                }

                pshm.nrecv = 0;
            }

            (*pstat).MPI_SOURCE = src;
            (*pstat).MPI_TAG = tag;
        }

        MPI_SUCCESS
    }
}