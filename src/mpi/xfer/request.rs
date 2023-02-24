use crate::debug::DbgEntryExit;
use crate::debug_xfer;
use crate::shared::*;

macro_rules! DbgEnEx {
    ($name:literal) => {
        let _dbgEntryExit = DbgEntryExit::new(|s| debug_xfer!($name, "{s}"));
    };
}

#[derive(Clone, Copy)]
pub struct Request {
    pub buf: *mut c_void,
    pub stat: MPI_Status,
    pub comm: MPI_Comm,
    pub flag: i32,
    pub tag: i32,
    pub cnt: i32,
    pub rank: i32,
    pub isColl: bool,
    pub collRoot: i32,
}

impl Default for Request {
    fn default() -> Self {
        Self::new()
    }
}

impl Request {
    pub const fn new() -> Self {
        Request {
            buf: std::ptr::null_mut(),
            stat: MPI_Status::new(),
            comm: 0,
            flag: 0,
            tag: 0,
            cnt: 0,
            rank: 0,
            isColl: false,
            collRoot: -1
        }
    }

    pub fn test(req: *mut Self, pflag: &mut i32, pstat: Option<&mut MPI_Status>) -> MpiResult {
        DbgEnEx!("Test");

        let code = Context::progress();
        if let Err(code) = code {
            return Err(Context::err_handler().call(MPI_COMM_WORLD, code));
        }

        *pflag = 0;
        let r = &mut unsafe { *req };

        if r.flag != 0 {
            debug_xfer!("Test", "Find request with tag: {}, rank: {}", r.tag, r.rank);
            *pflag = 1;

            if let Some(stat) = pstat {
                stat.MPI_SOURCE = Context::comm().rank_unmap(r.comm, r.stat.MPI_SOURCE);
                stat.MPI_TAG = Context::comm().tag_unmap(r.comm, r.stat.MPI_TAG);
                stat.cnt = r.stat.cnt;
            }
            r.flag = 0;
            Context::shm().free_req(req);
        }

        Ok(())
    }

    pub fn wait(&mut self, pstat: Option<&mut MPI_Status>) -> MpiResult {
        DbgEnEx!("Wait");
        let mut flag = 0;
        if let Some(stat) = pstat {
            while flag == 0 {
                Self::test(self, &mut flag, Some(stat))?;
            }
        } else {
            while flag == 0 {
                Self::test(self, &mut flag, None)?;
            }
        }
        Ok(())
    }

    pub fn wait_all(reqs: &mut [*mut Request], pstat: &mut [MPI_Status]) -> MpiResult {
        DbgEnEx!("WaitAll");
        debug_assert!(reqs.len() == pstat.len());

        for i in pstat.iter_mut() {
            i.MPI_ERROR = MPI_ERR_PENDING as i32;
        }

        let mut flag = 0;
        let mut flags = 0;

        while flags != reqs.len() as i32 {
            for (stat, req) in pstat.iter_mut().zip(reqs.iter_mut()) {
                if stat.MPI_ERROR == MPI_ERR_PENDING as i32 {
                    let ret = Self::test(*req, &mut flag, Some(stat));
                    if let Err(code) = ret {
                        stat.MPI_ERROR = code as i32;
                        return Err(MPI_ERR_IN_STATUS);
                    }
                    if flag != 0 {
                        stat.MPI_ERROR = MPI_SUCCESS;
                        flags += 1;
                    }
                }
            }
        }
        Ok(())
    }
}
