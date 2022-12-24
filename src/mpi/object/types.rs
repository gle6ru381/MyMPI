use crate::debug_objs;
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::slice::{from_raw_parts, from_raw_parts_mut};

use crate::shared::*;

pub trait Typed {
    const ALIGN: usize = 32;
    fn into_mpi() -> i32;
}

impl Typed for i32 {
    const ALIGN: usize = 32;
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
    const ALIGN: usize = 32;
    fn into_mpi() -> i32 {
        MPI_BYTE
    }
}

impl Typed for f64 {
    const ALIGN: usize = 32;
    fn into_mpi() -> i32 {
        MPI_DOUBLE
    }
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

impl<T: Typed> std::ops::Deref for Data<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.into_slice()
    }
}

impl<T: Typed> std::ops::DerefMut for Data<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.into_slice_mut()
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

    pub fn into_slice<'a>(&'a self) -> &'a [T] {
        unsafe { from_raw_parts(self.data, self.size) }
    }

    pub fn into_slice_mut<'a>(&'a mut self) -> &'a mut [T] {
        unsafe { from_raw_parts_mut(self.data, self.size) }
    }
}

pub struct Promise<'a, T: Typed + 'a> {
    req: MPI_Request,
    phantom: PhantomData<&'a T>,
}

impl<T: Typed> Drop for Promise<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.get_req().is_null() {
            self.call_wait();
        }
        debug_objs!("Promise", "Destroy");
    }
}

impl<T: Typed> Promise<'_, T> {
    pub(crate) const fn new(req: MPI_Request) -> Self {
        Promise {
            req,
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn call_wait(&self) -> i32 {
        debug_objs!("Promise", "Call wait");
        P_MPI_Request::wait(&mut self.get_req(), null_mut())
    }

    const fn get_req(&self) -> MPI_Request {
        self.req
    }

    #[must_use]
    pub fn request(&mut self) -> MPI_Request {
        std::mem::replace(&mut self.get_req(), null_mut())
    }

    pub(crate) fn release(&mut self) {
        self.req = null_mut();
    }

    pub fn wait(mut self) -> Result<Self, i32> {
        debug_assert!(!self.get_req().is_null());
        let res = self.call_wait();
        if res == MPI_SUCCESS {
            self.release();
            return Ok(self);
        }

        Err(res)
    }
}
