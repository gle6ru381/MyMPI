use crate::{shm::ShmData, p_mpi_errh_init, p_mpi_comm_init, MPI_CHECK, MPI_CHECK_COMM, MPI_CHECK_RANK, MPI_CHECK_TYPE, p_mpi_get_grank, p_mpi_get_gtag};
use std::{ffi::CStr, slice::from_raw_parts_mut};
pub use crate::types::*;
use crate::private::*;

pub struct Context {
    shm : ShmData,
    mpi_size : i32,
    mpi_rank : i32,
    mpi_init : bool
}

static mut context : Context = Context{shm: ShmData::new(), mpi_size: 1, mpi_rank: 0, mpi_init: false};

impl Context {
    fn get_env() {
        let size = std::env::var("MPI_SIZE").unwrap_or(String::from("")).parse::<i32>().unwrap_or(0);
        debug_assert!(size > 0);
        unsafe {context.mpi_size = size};
    }

    fn parseArgs(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32
    {
        unsafe {
            let mut n : i32 = context.mpi_size;
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
                            libc::EINVAL | libc::ERANGE => n = context.mpi_size,
                            _ => {
                                if n > 0 {
                                    *pargc -= 2;
                                    while false {
                                        argv = argv.add(1);
                                        *((*pargv).add((*pargc - argc) as usize)) = *argv;
                                        *argv = std::ptr::null_mut();
                                    }

                                    println!("{}/{}: n = {n}", context.mpi_rank, context.mpi_size);
                                } else {
                                    n = context.mpi_size;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            println!("Set mpi size on: {n}");
            context.mpi_size = n;
        }
        println!("Exit parse args");
        MPI_SUCCESS
    }

    pub fn split_proc(rank : i32, size : i32) -> i32 {
        unsafe {
            debug_assert!(!context.mpi_init);
            context.mpi_rank = rank
        };

        if size > 1 {
            unsafe {
                match libc::fork() {
                    -1 => return !MPI_SUCCESS,
                    0 => return Self::split_proc(rank + size / 2 + size % 2, size / 2),
                    _ => return Self::split_proc(rank, size / 2 + size % 2)
                }
            }
        }

        unsafe {
            println!("{}/{}: mpi rank = {}", context.mpi_rank, context.mpi_size, context.mpi_rank);
        }

        MPI_SUCCESS
    }

    pub fn init(pargc : *mut i32, pargv : *mut*mut*mut i8) -> i32 {
        unsafe {
            debug_assert!(!context.mpi_init);

            let mut code = p_mpi_errh_init(pargc, pargv);
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
            }

            Self::get_env();

            code = context.shm.init(pargc, pargv);
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
            }

            code = Self::split_proc(0, context.mpi_size);
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
            }

            code = p_mpi_comm_init(pargc, pargv);
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
            }

            context.mpi_init = true;
        }

        MPI_SUCCESS
    }

    pub fn deinit() -> i32 {
        unsafe {
            debug_assert!(context.mpi_init);
            context.mpi_init = false;
        }
        MPI_SUCCESS
    }

    #[inline(always)]
    pub fn size() -> i32 {
        unsafe {context.mpi_size}
    }

    #[inline(always)]
    pub fn rank() -> i32 {
        unsafe {context.mpi_rank}
    }

    #[inline(always)]
    pub fn is_init() -> bool {
        unsafe {context.mpi_init}
    }

    pub fn send(buf : *const c_void, mut cnt : i32, dtype : MPI_Datatype, mut dest : i32, mut tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
        MPI_CHECK!(Self::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
        MPI_CHECK_COMM!(comm);
        MPI_CHECK!(!buf.is_null(), comm, MPI_ERR_BUFFER);
        MPI_CHECK!(!preq.is_null(), comm, MPI_ERR_REQUEST);
        MPI_CHECK!(cnt >= 0, comm, MPI_ERR_COUNT);
        MPI_CHECK_TYPE!(dtype, comm);
        MPI_CHECK_RANK!(dest, comm);

        cnt *= p_mpi_type_size(dtype);
        dest = p_mpi_get_grank(comm, dest);

        MPI_CHECK!(dest != Self::rank(), comm, MPI_ERR_INTERN);
        MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_TAG);

        tag = p_mpi_get_gtag(comm, tag);

        println!("Send call to {dest} with tag {tag}");

        let code = unsafe {context.shm.progress()};
        if code != MPI_SUCCESS {
            return p_mpi_call_errhandler(comm, code);
        }

        unsafe {
            let oreq = context.shm.get_send();
            if oreq.is_some() {
                *preq = oreq.unwrap_unchecked();
                let req = &mut **preq;
                req.buf = buf as *mut c_void;
                req.cnt = cnt;
                req.rank = dest;
                req.tag = tag;
                req.comm = comm;
                req.flag = 0;

                return MPI_SUCCESS;
            } else {
                *preq = null_mut();
            }
        }
        p_mpi_call_errhandler(comm, MPI_ERR_INTERN)
    }

    pub fn recv(buf : *mut c_void, mut cnt : i32, dtype : MPI_Datatype, mut src: i32, mut tag : i32, comm : MPI_Comm, preq : *mut MPI_Request) -> i32 {
        MPI_CHECK!(Self::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
        MPI_CHECK_COMM!(comm);
        MPI_CHECK!(!buf.is_null(), comm, MPI_ERR_BUFFER);
        MPI_CHECK!(!preq.is_null(), comm, MPI_ERR_REQUEST);
        MPI_CHECK!(cnt >= 0, comm, MPI_ERR_COUNT);
        MPI_CHECK_TYPE!(dtype, comm);
        MPI_CHECK_RANK!(src, comm);

        cnt *= p_mpi_type_size(dtype);
        src = p_mpi_get_grank(comm, src);

        MPI_CHECK!(src != Self::rank(), comm, MPI_ERR_INTERN);
        MPI_CHECK!(tag >= 0 && tag <= 32767, comm, MPI_ERR_TAG);

        tag = p_mpi_get_gtag(comm, tag);

        println!("Recv call from {src} with tag {tag}");

        unsafe {
            let code = context.shm.progress();
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(comm, code);
            }

            *preq = context.shm.find_unexp(src, tag);
            println!("Find unexp: {}", !(*preq).is_null());
            if !(*preq).is_null() {
                let req = &mut **preq;
                if req.cnt > cnt {
                    return p_mpi_call_errhandler(comm, MPI_ERR_TRUNCATE);
                }

                buf.copy_from(req.buf, req.cnt as usize);
                let layout = std::alloc::Layout::from_size_align(req.cnt as usize, 1).unwrap();
                std::alloc::dealloc(req.buf as *mut u8, layout);
                req.buf = buf;
                req.comm = comm;
                req.stat.MPI_SOURCE = req.rank;
                req.stat.MPI_TAG = req.tag;
                req.stat.cnt = req.cnt;
                req.flag = 1;

                return MPI_SUCCESS;
            } else {
                let oreq = context.shm.get_recv();
                if oreq.is_some() {
                    *preq = oreq.unwrap_unchecked();
                    let req = &mut **preq;
                    
                    req.buf = buf;
                    req.cnt = cnt;
                    req.rank = src;
                    req.tag = tag;
                    req.comm = comm;
                    req.flag = 0;

                    return MPI_SUCCESS;
                } else {
                    *preq = null_mut();
                }
            }
        }

        p_mpi_call_errhandler(comm, MPI_ERR_INTERN)
    }

    pub fn test(preq : *mut MPI_Request, pflag : *mut i32, pstat : *mut MPI_Status) -> i32 {
        MPI_CHECK!(Self::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
        MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_REQUEST);
        MPI_CHECK!(!pflag.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

        unsafe {
            let code = context.shm.progress();
            if code != MPI_SUCCESS {
                return p_mpi_call_errhandler(MPI_COMM_WORLD, code);
            }
            *pflag = 0;
            if !(*preq).is_null() {
                let req = &mut **preq;

                if req.flag != 0 {
                    *pflag = 1;

                    if !pstat.is_null() {
                        let stat = &mut *pstat;
                        stat.MPI_SOURCE = req.stat.MPI_SOURCE;
                        stat.MPI_TAG = req.stat.MPI_TAG;
                        stat.cnt = req.stat.cnt;
                    }
                    req.flag = 0;
                    context.shm.free_req(req, *preq);
                }
            }
        }
        MPI_SUCCESS
    }

    pub fn wait(preq : *mut MPI_Request, pstat : *mut MPI_Status) -> i32 {
        let mut flag = 0;

        while flag == 0 {
            let code = Self::test(preq, &mut flag, pstat);
            if code != MPI_SUCCESS {
                return code;
            }
        }

        MPI_SUCCESS
    }

    pub fn wait_all(cnt : i32, preq : *mut MPI_Request, pstat : *mut MPI_Status) -> i32 {
        MPI_CHECK!(cnt >= 0, MPI_COMM_WORLD, MPI_ERR_COUNT);
        MPI_CHECK!(!pstat.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

        let stat = unsafe {from_raw_parts_mut(pstat, cnt as usize)};
        for i in &mut stat[..] {
            i.MPI_ERROR = MPI_ERR_PENDING;
        }

        let mut flag = 0;
        let mut flags = 0;

        while flags != cnt {
            for (i, s) in stat.iter_mut().enumerate() {
                if s.MPI_ERROR == MPI_ERR_PENDING {
                    let code = unsafe {Self::test(preq.add(i), &mut flag, pstat.add(i))};
                    if code != MPI_SUCCESS {
                        stat[i].MPI_ERROR = code;
                        return MPI_ERR_IN_STATUS;
                    }

                    if flag != 0 {
                        s.MPI_ERROR = MPI_SUCCESS;
                        flags += 1;
                    }
                }
            }
        }
        
        MPI_SUCCESS
    }
}