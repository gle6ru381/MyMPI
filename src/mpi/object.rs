use std::alloc::{alloc, dealloc, Layout};
use std::ops::Deref;
use std::slice::from_raw_parts_mut;

use crate::Buffer;
use crate::{private::*, Typed};

pub struct MpiObject {
    error: i32,
}

pub struct Data<T: Typed> {
    data: *mut T,
    size: usize,
}

impl<T: Typed> Drop for Data<T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                dealloc(
                    self.data as *mut u8,
                    Layout::from_size_align_unchecked(self.size * T::into_mpi() as usize, T::ALIGN),
                );
            }
        }
    }
}

impl<T: Typed> Data<T> {
    pub fn new(size: usize) -> Self {
        Data {
            data: unsafe {
                alloc(Layout::from_size_align_unchecked(
                    size * T::into_mpi() as usize,
                    T::ALIGN,
                )) as *mut T
            },
            size,
        }
    }

    pub const fn empty() -> Self {
        Data {
            data: null_mut(),
            size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_null()
    }

    pub const fn len(&self) -> usize {
        self.size
    }

    pub const fn raw(&self) -> *mut T {
        self.data
    }

    pub fn into_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe { from_raw_parts_mut(self.data, self.size) }
    }
}

pub struct Promise<T: Typed> {
    data: Data<T>,
    req: MPI_Request,
}

impl<T: Typed> Drop for Promise<T> {
    fn drop(&mut self) {
        if !self.req.is_null() {
            P_MPI_Request::wait(&mut self.req, null_mut());
        }
    }
}

impl Drop for MpiObject {
    fn drop(&mut self) {
        Context::deinit();
    }
}

impl<T: Typed> Promise<T> {
    #[allow(dead_code)]
    fn new_alloc(size: usize, req: MPI_Request) -> Self {
        Promise {
            data: Data::new(size),
            req,
        }
    }

    const fn new(req: MPI_Request) -> Self {
        Promise {
            data: Data::empty(),
            req,
        }
    }

    const fn from_data(data: Data<T>, req: MPI_Request) -> Self {
        Promise { data, req }
    }

    pub fn wait(&mut self) -> i32 {
        debug_assert!(!self.req.is_null());
        let res = P_MPI_Request::wait(&mut self.req, null_mut());
        if res == MPI_SUCCESS {
            self.req = null_mut();
        }
        res
    }

    pub fn data<'a>(&'a mut self) -> &'a mut [T] {
        debug_assert!(!self.data.is_empty());
        self.wait();
        self.data.into_slice()
    }

    pub fn into_data(&mut self) -> Data<T> {
        std::mem::replace(&mut self.data, Data::empty())
    }
}

impl MpiObject {
    pub fn new() -> Self {
        Context::init(null_mut(), null_mut());
        MpiObject { error: MPI_SUCCESS }
    }

    pub fn rank() -> i32 {
        debug_assert!(Context::is_init());
        Context::rank()
    }

    pub fn size() -> i32 {
        debug_assert!(Context::is_init());
        Context::size()
    }

    pub const fn last_error(&self) -> i32 {
        self.error
    }

    pub fn send_raw<'a, T: Typed>(
        &mut self,
        buff: &'a [T],
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> bool {
        debug!("Enter send raw");
        let opromise = self.send_req_raw(buff, rank, tag, comm);
        if opromise.is_none() {
            return false;
        }
        let mut promise = unsafe { opromise.unwrap_unchecked() };
        self.error = promise.wait();

        debug!("Exit send raw");

        self.error == MPI_SUCCESS
    }

    pub fn send<'a, T: Typed, A: Deref<Target = [T]>>(
        &'a mut self,
        buff: &'a A,
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> bool {
        self.send_raw(buff.deref(), rank, tag, comm)
    }

    #[must_use]
    pub fn send_req_raw<'a, T: Typed>(
        &mut self,
        buff: &'a [T],
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> Option<Promise<T>> {
        let mut req: MPI_Request = uninit();
        self.error = Buffer::from_type(buff, T::into_mpi())
            .set_rank(rank)
            .set_tag(tag)
            .set_comm(comm)
            .send(&mut req);
        if self.error != MPI_SUCCESS {
            None
        } else {
            Some(Promise::new(req))
        }
    }

    #[must_use]
    pub fn send_req<'a, T: Typed, A: Deref<Target = [T]>>(
        &'a mut self,
        buff: &'a A,
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> Option<Promise<T>> {
        self.send_req_raw(buff.deref(), rank, tag, comm)
    }

    #[must_use]
    pub fn recv_req<T: Typed>(
        &mut self,
        size: usize,
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> Option<Promise<T>> {
        let mut req: MPI_Request = uninit();
        let mut data = Data::new(size);
        self.error = Buffer::from_type_mut(data.into_slice(), T::into_mpi())
            .set_rank(rank)
            .set_tag(tag)
            .set_comm(comm)
            .recv(&mut req);
        if self.error != MPI_SUCCESS {
            None
        } else {
            Some(Promise::from_data(data, req))
        }
    }

    #[must_use]
    pub fn recv<T: Typed>(
        &mut self,
        size: usize,
        rank: i32,
        tag: i32,
        comm: MPI_Comm,
    ) -> Option<Data<T>> {
        let opromise = self.recv_req(size, rank, tag, comm);
        if opromise.is_none() {
            return None;
        }
        let mut promise = unsafe { opromise.unwrap_unchecked() };
        self.error = promise.wait();
        if self.error != MPI_SUCCESS {
            return None;
        }
        Some(promise.into_data())
    }
}
