use crate::context::Context;
use crate::{private::*, MPI_CHECK};
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub struct Buffer {
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    rank: i32,
    tag: i32,
    comm: MPI_Comm,
}

pub trait Typed {
    const ALIGN: usize = 1;
    fn into_mpi() -> i32;
}

impl Typed for i32 {
    const ALIGN: usize = 4;
    fn into_mpi() -> i32 {
        MPI_INT
    }
}

impl Typed for i8 {
    fn into_mpi() -> i32 {
        MPI_BYTE
    }
}

impl Typed for u8 {
    const ALIGN: usize = 1;
    fn into_mpi() -> i32 {
        MPI_BYTE
    }
}

impl Typed for f64 {
    const ALIGN: usize = 8;
    fn into_mpi() -> i32 {
        MPI_DOUBLE
    }
}

impl P_MPI_Request {
    pub fn test(preq: &mut MPI_Request, pflag: &mut i32, pstat: *mut MPI_Status) -> i32 {
        let code = Context::progress();
        if code != MPI_SUCCESS {
            return Context::err_handler().call(MPI_COMM_WORLD, code);
        }
        *pflag = 0;
        if !(*preq).is_null() {
            let req = unsafe { &mut **preq };
            if req.flag != 0 {
                debug!("Find request with tag: {}, rank: {}", req.tag, req.rank);
                *pflag = 1;

                if !pstat.is_null() {
                    let stat = unsafe { &mut *pstat };
                    stat.MPI_SOURCE = Context::comm().rank_unmap(req.comm, req.stat.MPI_SOURCE);
                    stat.MPI_TAG = Context::comm().tag_unmap(req.comm, req.stat.MPI_TAG);
                    stat.cnt = req.stat.cnt;
                }
                req.flag = 0;
                Context::shm().free_req(req, *preq);
            }
        }
        MPI_SUCCESS
    }

    pub fn wait(preq: &mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
        let mut flag = 0;
        debug!("Call wait");
        while flag == 0 {
            let code = Self::test(preq, &mut flag, pstat);
            if code != MPI_SUCCESS {
                return code;
            }
        }
        MPI_SUCCESS
    }

    pub fn wait_all(reqs: &mut [MPI_Request], pstat: &mut [MPI_Status]) -> i32 {
        debug_assert!(reqs.len() == pstat.len());
        debug!("Wait all, size: {}", reqs.len());

        for i in pstat.iter_mut() {
            i.MPI_ERROR = MPI_ERR_PENDING;
        }

        let mut flag = 0;
        let mut flags = 0;

        while flags != reqs.len() as i32 {
            for (stat, req) in pstat.iter_mut().zip(reqs.iter_mut()) {
                if stat.MPI_ERROR == MPI_ERR_PENDING {
                    let code = Self::test(req, &mut flag, stat);
                    if code != MPI_SUCCESS {
                        stat.MPI_ERROR = code;
                        return MPI_ERR_IN_STATUS;
                    }
                    if flag != 0 {
                        stat.MPI_ERROR = MPI_SUCCESS;
                        flags += 1;
                    }
                }
            }
        }
        MPI_SUCCESS
    }
}

impl Buffer {
    fn to_request(&self, flag: i32) -> P_MPI_Request {
        P_MPI_Request {
            buf: self.buf,
            stat: MPI_Status::new(),
            comm: self.comm,
            flag,
            tag: Context::comm().tag_map(self.comm, self.tag),
            cnt: self.bytes(),
            rank: Context::comm().rank_map(self.comm, self.rank),
        }
    }

    pub const fn new() -> Self {
        Buffer {
            buf: null_mut(),
            cnt: 0,
            dtype: 0,
            rank: 0,
            tag: 0,
            comm: 0,
        }
    }

    pub const fn from_type<'a, T: Typed>(data: &'a [T], dtype: i32) -> Self {
        Buffer {
            buf: data.as_ptr() as *mut c_void,
            cnt: data.len() as i32,
            dtype,
            rank: 0,
            tag: 0,
            comm: 0,
        }
    }

    pub fn from_type_mut<'a, T: Typed>(data: &'a mut [T], dtype: i32) -> Self {
        Buffer {
            buf: data.as_ptr() as *mut c_void,
            cnt: data.len() as i32,
            dtype,
            rank: 0,
            tag: 0,
            comm: 0,
        }
    }

    #[must_use]
    pub fn from<'a, T: Typed>(data: &'a [T]) -> Self {
        Buffer {
            buf: data.as_ptr() as *mut c_void,
            cnt: data.len() as i32,
            dtype: T::into_mpi(),
            rank: 0,
            tag: 0,
            comm: 0,
        }
    }

    #[must_use]
    pub fn from_mut<'a, T: Typed>(data: &'a mut [T]) -> Self {
        Buffer {
            buf: data.as_mut_ptr() as *mut c_void,
            cnt: data.len() as i32,
            dtype: T::into_mpi(),
            rank: 0,
            tag: 0,
            comm: 0,
        }
    }

    #[must_use]
    pub fn set_data<'a, D: Typed, T: Deref<Target = [D]>>(self, data: &'a T) -> Self {
        self.set_data_raw(data.deref())
    }

    #[must_use]
    pub fn set_data_mut<'a, D: Typed, T: DerefMut<Target = [D]>>(self, data: &'a mut T) -> Self {
        self.set_data_raw_mut(&mut data.deref_mut())
    }

    #[must_use]
    pub fn set_data_raw<'a, T: Typed>(mut self, data: &'a [T]) -> Self {
        self.buf = data.as_ptr() as *mut c_void;
        self.dtype = T::into_mpi();
        self.cnt = data.len() as i32;
        self
    }

    #[must_use]
    pub fn set_data_raw_mut<'a, T: Typed>(mut self, data: &'a mut [T]) -> Self {
        self.buf = data.as_mut_ptr() as *mut c_void;
        self.dtype = T::into_mpi();
        self.cnt = data.len() as i32;
        self
    }

    #[must_use]
    pub fn set_rank(mut self, rank: i32) -> Self {
        self.rank = rank;
        self
    }

    #[must_use]
    pub fn set_tag(mut self, tag: i32) -> Self {
        self.tag = tag;
        self
    }

    #[must_use]
    pub fn set_comm(mut self, comm: MPI_Comm) -> Self {
        self.comm = comm;
        self
    }

    pub fn data<'a, T: Typed>(&'a self) -> &'a [T] {
        debug_assert_eq!(T::into_mpi(), self.dtype);
        debug!("Get data: {:X}", self.buf as usize);
        unsafe { from_raw_parts(self.buf as *const T, self.cnt as usize) }
    }

    pub unsafe fn data_mut_unchecked<'a, T: Typed>(&'a mut self, size: usize) -> &'a mut [T] {
        from_raw_parts_mut(self.buf as *mut T, size as usize)
    }

    pub fn data_mut<'a, T: Typed>(&'a mut self, size: usize) -> &'a mut [T] {
        debug_assert_eq!(T::into_mpi(), self.dtype);
        debug_assert!(size < self.bytes() as usize);
        unsafe { self.data_mut_unchecked(size) }
    }

    pub fn rank(&self) -> i32 {
        self.rank
    }

    pub fn tag(&self) -> i32 {
        self.tag
    }

    pub fn comm(&self) -> MPI_Comm {
        self.comm
    }

    pub fn bytes(&self) -> i32 {
        self.cnt * p_mpi_type_size(self.dtype)
    }

    pub fn send(&self, req: &mut MPI_Request) -> i32 {
        let dest = Context::comm().rank_map(self.comm, self.rank);

        MPI_CHECK!(dest != Context::rank(), self.comm, MPI_ERR_INTERN);
        MPI_CHECK!(self.tag >= 0 && self.tag <= 32767, self.comm, MPI_ERR_TAG);

        let tag = Context::comm().tag_map(self.comm, self.tag);
        debug!("Send call to {dest} with tag {tag}");

        let code = Context::progress();
        if code != MPI_SUCCESS {
            return Context::err_handler().call(self.comm, code);
        }

        let new_req = Context::shm().get_send();
        if new_req.is_some() {
            *req = unsafe { new_req.unwrap_unchecked() };
            unsafe {
                (**req) = self.to_request(0);
            }
            return MPI_SUCCESS;
        } else {
            *req = null_mut();
        }
        Context::err_handler().call(self.comm, MPI_ERR_INTERN)
    }

    pub fn recv(&self, req: &mut MPI_Request) -> i32 {
        let src = Context::comm().rank_map(self.comm, self.rank);

        MPI_CHECK!(self.rank != Context::rank(), self.comm, MPI_ERR_INTERN);
        MPI_CHECK!(
            self.tag >= 0 && self.tag <= 32767,
            self.comm,
            MPI_ERR_INTERN
        );

        let tag = Context::comm().tag_map(self.comm, self.tag);

        debug!("Recv call from {src} with tag {tag}");

        let code = Context::progress();
        if code != MPI_SUCCESS {
            return Context::err_handler().call(self.comm, code);
        }

        *req = Context::shm().find_unexp(src, tag);
        if !(*req).is_null() {
            let rreq = unsafe { &mut **req };
            debug!("Unexprect rank: {}, tag: {}", rreq.rank, rreq.tag);
            if rreq.cnt > self.bytes() {
                debug!("Error truncate unexpect");
                return Context::err_handler().call(self.comm, MPI_ERR_TRUNCATE);
            }

            unsafe {
                self.buf.copy_from(rreq.buf, rreq.cnt as usize);
                let layout = std::alloc::Layout::from_size_align_unchecked(rreq.cnt as usize, 1);
                std::alloc::dealloc(rreq.buf as *mut u8, layout);
            }
            *rreq = self.to_request(1);
            return MPI_SUCCESS;
        } else {
            debug!("Generate new request");
            let rreq = Context::shm().get_recv();
            if rreq.is_some() {
                unsafe {
                    *req = rreq.unwrap_unchecked();
                    **req = self.to_request(0);
                }
                return MPI_SUCCESS;
            } else {
                *req = null_mut();
            }
        }

        Context::err_handler().call(self.comm, MPI_ERR_INTERN)
    }
}

#[no_mangle]
pub extern "C" fn MPI_Send(
    buf: *const c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    dest: i32,
    tag: i32,
    comm: MPI_Comm,
) -> i32 {
    let mut req: MPI_Request = uninit();
    let mut code = MPI_Isend(buf, cnt, dtype, dest, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    println!("Send isend finish");
    code = MPI_Wait(&mut req, null_mut());
    if code != MPI_SUCCESS {
        return code;
    }

    println!("Send finish");

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Recv(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    src: i32,
    tag: i32,
    comm: MPI_Comm,
    pstat: *mut MPI_Status,
) -> i32 {
    let mut req: MPI_Request = uninit();
    let mut code = MPI_Irecv(buf, cnt, dtype, src, tag, comm, &mut req);
    if code != MPI_SUCCESS {
        return code;
    }
    println!("Recv irecv finish");
    code = MPI_Wait(&mut req, pstat);
    if code != MPI_SUCCESS {
        return code;
    }

    println!("Recv finish");

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Sendrecv(
    sbuf: *const c_void,
    scnt: i32,
    sdtype: MPI_Datatype,
    dest: i32,
    stag: i32,
    rbuf: *mut c_void,
    rcnt: i32,
    rdtype: MPI_Datatype,
    src: i32,
    rtag: i32,
    comm: MPI_Comm,
    pstat: *mut MPI_Status,
) -> i32 {
    MPI_CHECK!(Context::is_init(), comm, MPI_ERR_OTHER);
    MPI_CHECK!(!pstat.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    let mut req: [MPI_Request; 2] = uninit();
    let mut stat: [MPI_Status; 2] = uninit();

    CHECK_RET!(MPI_Isend(sbuf, scnt, sdtype, dest, stag, comm, &mut req[0]));
    CHECK_RET!(MPI_Irecv(rbuf, rcnt, rdtype, src, rtag, comm, &mut req[1]));
    CHECK_RET!(MPI_Waitall(2, req.as_mut_ptr(), stat.as_mut_ptr()));

    unsafe {
        pstat.write(stat[1]);
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Isend(
    buf: *const c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    dest: i32,
    tag: i32,
    comm: MPI_Comm,
    preq: *mut MPI_Request,
) -> i32 {
    Buffer {
        buf: buf as *mut c_void,
        cnt,
        dtype,
        rank: dest,
        tag,
        comm,
    }
    .send(unsafe { &mut *preq })
}

#[no_mangle]
pub extern "C" fn MPI_Irecv(
    buf: *mut c_void,
    cnt: i32,
    dtype: MPI_Datatype,
    src: i32,
    tag: i32,
    comm: MPI_Comm,
    preq: *mut MPI_Request,
) -> i32 {
    Buffer {
        buf,
        cnt,
        dtype,
        rank: src,
        tag,
        comm,
    }
    .recv(unsafe { &mut *preq })
}

#[no_mangle]
pub extern "C" fn MPI_Test(preq: *mut MPI_Request, pflag: *mut i32, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pflag.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    P_MPI_Request::test(unsafe { &mut *preq }, unsafe { &mut *pflag }, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Wait(preq: *mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(!preq.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    P_MPI_Request::wait(unsafe { &mut *preq }, pstat)
}

#[no_mangle]
pub extern "C" fn MPI_Waitall(cnt: i32, preq: *mut MPI_Request, pstat: *mut MPI_Status) -> i32 {
    MPI_CHECK!(
        !preq.is_null() && !pstat.is_null(),
        MPI_COMM_WORLD,
        MPI_ERR_ARG
    );
    MPI_CHECK!(cnt >= 0, MPI_COMM_WORLD, MPI_ERR_COUNT);
    if cnt == 0 {
        return MPI_SUCCESS;
    }
    P_MPI_Request::wait_all(
        unsafe { std::slice::from_raw_parts_mut(preq, cnt as usize) },
        unsafe { std::slice::from_raw_parts_mut(pstat, cnt as usize) },
    )
}
