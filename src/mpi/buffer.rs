use std::{alloc::Layout, slice::from_raw_parts_mut, ptr::null_mut};

use libc::c_void;
use crate::types::MPI_Datatype;

pub struct DynBuffer {
    data: *mut u8,
    layout: Layout
}

impl Drop for DynBuffer {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {std::alloc::dealloc(self.data, self.layout)};
        }
    }
}

impl DynBuffer {
    pub fn empty() -> Self {
        Self {
            data: null_mut(),
            layout: Layout::new::<u8>()
        }
    }

    pub fn new(size: usize) -> Self {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, 16);
            let data = std::alloc::alloc(layout);
            DynBuffer { data, layout }
        }
    }

    pub const fn ptr(&self) -> *mut u8 {
        self.data
    }

    pub fn to_slice<'a>(&'a self) -> &'a mut [u8] {
        unsafe {from_raw_parts_mut(self.data, self.layout.size())}
    }
}