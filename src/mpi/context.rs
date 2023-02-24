use crate::backend::shm::ShmData;
use crate::communicator::group::CommGroup;
use crate::debug_core;
use crate::errhandler::handler::HandlerContext;
pub use crate::shared::*;
pub use crate::types::*;
use crate::xfer::collectives::allgather::allgather_simple;
use crate::xfer::collectives::allgather::AllgatherFn;
use crate::xfer::collectives::allreduce::AllreduceFn;
use crate::xfer::collectives::barrier::BarrierFn;
use crate::xfer::collectives::bcast::BCastFn;
use crate::xfer::collectives::gather::GatherFn;
use crate::xfer::collectives::reduce::ReduceFn;
use crate::xfer::collectives::*;
use std::ffi::CStr;
use std::process::exit;

macro_rules! debug_init {
    ($fmt:literal) => {
        debug_core!("Initialization", $fmt);
    };
    ($($args:tt)*) => {
        debug_core!("Initialization", $($args)*);
    }
}

pub struct Context {
    shm: ShmData,
    err_handler: HandlerContext,
    comm_group: CommGroup,
    mpi_size: i32,
    mpi_rank: i32,
    mpi_init: bool,
    use_nt: bool,
    barrier_impl: BarrierFn,
    bcast_impl: BCastFn,
    reduce_impl: ReduceFn,
    gather_impl: GatherFn,
    allreduce_impl: AllreduceFn,
    allgather_impl: AllgatherFn,
}

static mut CONTEXT: Context = Context {
    shm: ShmData::new(),
    mpi_size: 1,
    mpi_rank: 0,
    mpi_init: false,
    err_handler: HandlerContext::new(),
    comm_group: CommGroup::new(),
    use_nt: false,
    barrier_impl: barrier::barrier_simple,
    bcast_impl: bcast::bcast_shm,
    reduce_impl: reduce::reduce_ring,
    gather_impl: gather::gather_ring,
    allreduce_impl: allreduce::allreduce_simple,
    allgather_impl: allgather_simple,
};

struct SlurmData {
    size: i32,
    rank: i32,
    key: i32,
}

unsafe extern "C" fn child_handler(_: i32) {
    let mut retcode = 0;
    if libc::waitpid(-1, &mut retcode, libc::WNOHANG) > 0 {
        if retcode != 0 {
            exit(-1);
        }
    }
}

impl Context {
    fn get_use_nt() -> bool {
        if cfg!(feature = "ntcpy") {
            return true;
        }
        if let Ok(val) = std::env::var("MPI_USE_NT") {
            return val == "1";
        }
        false
    }

    fn get_mpi() -> Option<i32> {
        let size_env = std::env::var("MPI_SIZE");
        if let Ok(size) = size_env {
            if let Ok(res) = size.parse() {
                return Some(res);
            }
        }
        None
    }

    fn get_slurm() -> Option<SlurmData> {
        let rank_env = std::env::var("SLURM_PROCID");
        if let Ok(rank) = rank_env {
            let size_env = std::env::var("SLURM_NTASKS_PER_NODE");
            if let Ok(size) = size_env {
                let size_i = size.parse();
                let rank_i = rank.parse();
                if size_i.is_ok() && rank_i.is_ok() {
                    let key = std::env::var("SLURM_JOBID")
                        .unwrap_or(String::from("-1"))
                        .parse()
                        .unwrap_or(-1);
                    return unsafe {
                        Some(SlurmData {
                            size: size_i.unwrap_unchecked(),
                            rank: rank_i.unwrap_unchecked(),
                            key,
                        })
                    };
                }
            }
            return Some(SlurmData {
                size: 1,
                rank: 0,
                key: 0,
            });
        }
        None
    }

    fn get_env() -> i32 {
        unsafe {
            CONTEXT.use_nt = Self::get_use_nt();
            if CONTEXT.use_nt {
                debug_init!("Enable non-temporal copy");
            } else {
                debug_init!("Disable non-temporal copy");
            }
            if let Some(size) = Self::get_mpi() {
                CONTEXT.mpi_size = size;
                CONTEXT.mpi_rank = -1;
                debug_init!("Using internal, size: {size}");
                return -1;
            } else if let Some(data) = Self::get_slurm() {
                CONTEXT.mpi_size = data.size;
                CONTEXT.mpi_rank = data.rank;
                debug_init!("Using slurm, size: {}, rank: {}", data.size, data.rank);
                debug_assert!(data.key >= 0);
                return data.key;
            } else {
                panic!();
            }
        }
    }

    pub fn use_nt() -> bool {
        unsafe { CONTEXT.use_nt }
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

    pub fn progress() -> MpiResult {
        debug_core!("Progress", "Enter");
        let ret = unsafe { CONTEXT.shm.progress() };
        if let Err(code) = ret {
            debug_core!("Progress", "Exit with error: {}", code as i32);
            return Err(code);
        }
        debug_core!("Progress", "Exit success");
        Ok(())
    }

    pub fn call_error(comm: MPI_Comm, code: MpiError) {
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
                argv = argv.add(1);
                let mut val = *argv;
                while *val != '\0' as i8 {
                    val = val.add(1);
                }
                let tmp = CStr::from_ptr(*argv);
                let mut arg = tmp.to_str().unwrap();
                if arg == "-n" || arg == "-np" {
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
                                        *((*pargv).add((*pargc - argc) as usize)) = *argv;
                                        *argv = std::ptr::null_mut();
                                    }
                                } else {
                                    n = CONTEXT.mpi_size;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            CONTEXT.mpi_size = n;
        }
        MPI_SUCCESS
    }

    pub fn split_proc(rank: i32, size: i32) -> MpiResult {
        unsafe {
            debug_assert!(!CONTEXT.mpi_init);
            CONTEXT.mpi_rank = rank
        };

        if size > 1 {
            unsafe {
                match libc::fork() {
                    -1 => return Err(MPI_ERR_OTHER),
                    0 => return Self::split_proc(rank + size / 2 + size % 2, size / 2),
                    _ => return Self::split_proc(rank, size / 2 + size % 2),
                }
            }
        }

        Ok(())
    }

    pub fn init(pargc: *mut i32, pargv: *mut *mut *mut i8) -> MpiResult {
        unsafe {
            debug_assert!(!CONTEXT.mpi_init);

            let key = Self::get_env();

            let mut code = CONTEXT.shm.init(pargc, pargv, key);
            if let Err(code) = code {
                debug_init!("Error shm init");
                return Err(CONTEXT.err_handler.call(MPI_COMM_WORLD, code));
            }

            if CONTEXT.mpi_rank == -1 {
                code = Self::split_proc(0, CONTEXT.mpi_size);
                if let Err(code) = code {
                    debug_init!("Error split processors");
                    return Err(CONTEXT.err_handler.call(MPI_COMM_WORLD, code));
                }
                libc::signal(libc::SIGCHLD, child_handler as usize);
            }

            code = CONTEXT.comm_group.init(pargc, pargv);
            if let Err(code) = code {
                debug_init!("Error init communicators");
                return Err(CONTEXT.err_handler.call(MPI_COMM_WORLD, code));
            }

            CONTEXT.mpi_init = true;
        }

        debug_init!("Success");

        Ok(())
    }

    pub fn deinit() -> MpiResult {
        debug_init!("Begin finalize");
        unsafe {
            debug_assert!(CONTEXT.mpi_init);
            (CONTEXT.barrier_impl)(MPI_COMM_WORLD)?;
            CONTEXT.shm.deinit()?;
            CONTEXT.comm_group.deinit();
            CONTEXT.mpi_init = false;
            if CONTEXT.mpi_rank == 0 {
                libc::signal(libc::SIGCHLD, libc::SIG_IGN);
            } else {
                std::process::exit(0);
            }
        }
        Ok(())
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
    pub fn comm_size(comm: i32) -> i32 {
        unsafe { CONTEXT.comm_group.comm_size(comm) }
    }

    #[inline(always)]
    pub fn comm_rank(comm: i32) -> i32 {
        unsafe { CONTEXT.comm_group.comm_rank(comm) }
    }

    #[inline(always)]
    pub fn comm_prank(comm: i32, idx: i32) -> i32 {
        unsafe { CONTEXT.comm_group.comm_prank(comm, idx) }
    }

    #[inline(always)]
    pub fn is_init() -> bool {
        unsafe { CONTEXT.mpi_init }
    }

    #[inline(always)]
    pub fn barrier() -> BarrierFn {
        unsafe { CONTEXT.barrier_impl }
    }

    #[inline(always)]
    pub fn bcast() -> BCastFn {
        unsafe { CONTEXT.bcast_impl }
    }

    #[inline(always)]
    pub fn reduce() -> ReduceFn {
        unsafe { CONTEXT.reduce_impl }
    }

    #[inline(always)]
    pub fn gather() -> GatherFn {
        unsafe { CONTEXT.gather_impl }
    }

    #[inline(always)]
    pub fn allreduce() -> AllreduceFn {
        unsafe { CONTEXT.allreduce_impl }
    }

    #[inline(always)]
    pub fn allgather() -> AllgatherFn {
        unsafe { CONTEXT.allgather_impl }
    }
}
