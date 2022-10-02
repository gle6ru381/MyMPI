use std::alloc::Layout;
use std::alloc::{alloc, dealloc};
use std::mem::size_of;

use crate::{private::*, MPI_Waitall};
use crate::Buffer;
use crate::MPI_Wait;
pub struct MpiObject {
    reqs: Vec<MPI_Request>,
    error: i32,
}

impl Drop for MpiObject {
    fn drop(&mut self) {
        self.wait_all(None);
    }
}

impl MpiObject {
    pub const fn new() -> Self {
        MpiObject { reqs: Vec::new(), error: MPI_SUCCESS }
    }

    pub fn send(&mut self, buffer: &Buffer) -> usize {
        let mut req: MPI_Request = uninit();
        self.error = buffer.send(&mut req);
        self.reqs.push(req);
        self.reqs.len() - 1
    }

    pub fn recv(&mut self, buffer: &Buffer) -> usize {
        let mut req: MPI_Request = uninit();
        self.error = buffer.recv(&mut req);
        self.reqs.push(req);
        self.reqs.len() - 1
    }

    pub fn wait(&mut self, idx: usize) -> MPI_Status {
        let mut req = self.reqs[idx];
        let mut status = MaybeUninit::uninit();
        self.error = MPI_Wait(&mut req, status.as_mut_ptr());

        self.reqs.remove(idx);

        unsafe { status.assume_init() }
    }

    pub fn wait_all(&mut self, stats: Option<&mut [MPI_Status]>) -> bool {
        let pstats;
        let is_alloc;
        let layout = unsafe {
            Layout::from_size_align_unchecked(size_of::<MPI_Status>() * self.reqs.len(), 8)
        };

        if stats.is_some() {
            unsafe {
                let s = stats.unwrap_unchecked();
                debug_assert_eq!(s.len(), self.reqs.len()); 
                pstats = s.as_mut_ptr() 
            };
            is_alloc = false;
        } else {
            unsafe { pstats = alloc(layout) as *mut MPI_Status };
            is_alloc = true;
        }
        self.error = MPI_Waitall(self.reqs.len() as i32, self.reqs.as_mut_ptr(), pstats);
        if is_alloc {
            unsafe {dealloc(pstats as *mut u8, layout)}
        }
        true
    }

    pub fn last_error(&self) -> i32 {
        self.error
    }
}
