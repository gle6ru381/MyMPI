use crate::comm::CommGroup;
pub use crate::private::*;
use crate::shm::ShmData;
pub use crate::types::*;
use crate::{debug, HandlerContext};
use std::ffi::CStr;

// #[cfg(test)]
// use zstr::zstr;

pub struct Context {
    shm: ShmData,
    err_handler: HandlerContext,
    comm_group: CommGroup,
    mpi_size: i32,
    mpi_rank: i32,
    mpi_init: bool,
}

static mut CONTEXT: Context = Context {
    shm: ShmData::new(),
    mpi_size: 1,
    mpi_rank: 0,
    mpi_init: false,
    err_handler: HandlerContext::new(),
    comm_group: CommGroup::new(),
};

impl Context {
    fn get_env() {
        let size = std::env::var("MPI_SIZE")
            .unwrap_or(String::from(""))
            .parse::<i32>()
            .unwrap_or(0);
        debug_assert!(size > 0);
        unsafe { CONTEXT.mpi_size = size };
    }

    pub fn comm() -> &'static mut CommGroup {
        unsafe { &mut CONTEXT.comm_group }
    }

    pub fn err_handler() -> &'static mut HandlerContext {
        unsafe { &mut CONTEXT.err_handler }
    }

    pub fn shm() -> &'static mut ShmData {
        unsafe { &mut CONTEXT.shm }
    }

    pub fn progress() -> i32 {
        unsafe { CONTEXT.shm.progress() }
    }

    pub fn call_error(comm: MPI_Comm, code: i32) {
        unsafe {
            CONTEXT.err_handler.call(comm, code);
        }
    }

    #[allow(dead_code)]
    fn parse_args(pargc: *mut i32, pargv: *mut *mut *mut i8) -> i32 {
        unsafe {
            let mut n: i32 = CONTEXT.mpi_size;
            let argc = *pargc;
            let mut argv = *pargv;

            while --argc != 0 {
                println!("Arg step");
                argv = argv.add(1);
                println!("Arg: {:x}", *argv as usize);
                let mut val = *argv;
                while *val != '\0' as i8 {
                    println!("Val: {}", *val as u8 as char);
                    val = val.add(1);
                }
                let tmp = CStr::from_ptr(*argv);
                println!("Tmp: {}", tmp.as_ptr() as usize);
                let mut arg = tmp.to_str().unwrap();
                if arg == "-n" || arg == "-np" {
                    println!("Step in np");
                    if argc > 1 {
                        argv = argv.add(1);
                        arg = CStr::from_ptr(*argv).to_str().unwrap();
                        n = arg.parse::<i32>().unwrap();

                        let err = *libc::__errno_location();
                        match err {
                            libc::EINVAL | libc::ERANGE => n = CONTEXT.mpi_size,
                            _ => {
                                if n > 0 {
                                    *pargc -= 2;
                                    while !argv.is_null() {
                                        argv = argv.add(1);
                                        *pargc -= 1;
                                        println!("Decrement argc: {}", *pargc);
                                        *((*pargv).add((*pargc - argc) as usize)) = *argv;
                                        *argv = std::ptr::null_mut();
                                    }

                                    println!("{}/{}: n = {n}", CONTEXT.mpi_rank, CONTEXT.mpi_size);
                                } else {
                                    n = CONTEXT.mpi_size;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            println!("Set mpi size on: {n}");
            CONTEXT.mpi_size = n;
        }
        println!("Exit parse args");
        MPI_SUCCESS
    }

    pub fn split_proc(rank: i32, size: i32) -> i32 {
        unsafe {
            debug_assert!(!CONTEXT.mpi_init);
            CONTEXT.mpi_rank = rank
        };

        if size > 1 {
            unsafe {
                match libc::fork() {
                    -1 => return !MPI_SUCCESS,
                    0 => return Self::split_proc(rank + size / 2 + size % 2, size / 2),
                    _ => return Self::split_proc(rank, size / 2 + size % 2),
                }
            }
        }

        unsafe {
            println!(
                "{}/{}: mpi rank = {}",
                CONTEXT.mpi_rank, CONTEXT.mpi_size, CONTEXT.mpi_rank
            );
        }

        MPI_SUCCESS
    }

    pub fn init(pargc: *mut i32, pargv: *mut *mut *mut i8) -> i32 {
        unsafe {
            debug_assert!(!CONTEXT.mpi_init);

            Self::get_env();

            let mut code = CONTEXT.shm.init(pargc, pargv);
            if code != MPI_SUCCESS {
                CONTEXT.err_handler.call(MPI_COMM_WORLD, code);
            }

            code = Self::split_proc(0, CONTEXT.mpi_size);
            if code != MPI_SUCCESS {
                return CONTEXT.err_handler.call(MPI_COMM_WORLD, code);
            }

            code = CONTEXT.comm_group.init(pargc, pargv);
            if code != MPI_SUCCESS {
                return CONTEXT.err_handler.call(MPI_COMM_WORLD, code);
            }

            CONTEXT.mpi_init = true;
        }

        MPI_SUCCESS
    }

    pub fn deinit() -> i32 {
        unsafe {
            debug_assert!(CONTEXT.mpi_init);
            debug!("Context deinit");
            CONTEXT.shm.deinit();
            CONTEXT.mpi_init = false;
            if CONTEXT.mpi_rank == 0 {
                libc::signal(libc::SIGCHLD, libc::SIG_IGN);
            } else {
                std::process::exit(0);
            }
            debug!("Exit deinit");
        }
        MPI_SUCCESS
    }

    #[inline(always)]
    pub fn size() -> i32 {
        unsafe { CONTEXT.mpi_size }
    }

    #[inline(always)]
    pub fn rank() -> i32 {
        unsafe { CONTEXT.mpi_rank }
    }

    #[inline(always)]
    pub fn is_init() -> bool {
        unsafe { CONTEXT.mpi_init }
    }
}
