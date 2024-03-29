use super::types::Promise;
use super::types::Typed;
use crate::context::Context;
use crate::debug_objs;
use crate::shared::*;
use crate::xfer::ppp::{recv::*, send::*};
use crate::xfer::request::Request;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::null_mut;

pub struct MpiObject {}

pub struct Communicator {
    comm_id: MPI_Comm,
}

impl Drop for MpiObject {
    fn drop(&mut self) {
        debug_objs!("Initialization", "Finalize MPI");
        unsafe {Context::deinit().unwrap_unchecked()};
    }
}

impl MpiObject {
    pub fn new() -> MpiObject {
        debug_objs!("Initialization", "Begin init MPI");
        unsafe {Context::init(null_mut(), null_mut()).unwrap_unchecked()};
        debug_objs!("Initialization", "Finish init MPI");
        MpiObject {}
    }

    pub fn rank() -> i32 {
        Context::rank()
    }

    pub fn get_comm(&mut self, comm_id: MPI_Comm) -> Result<Communicator, MpiError> {
        Context::comm().check(comm_id)?;
        Ok(Communicator { comm_id })
    }
}

impl Communicator {
    pub fn send_slice<'a, T: Typed>(
        &self,
        buf: &'a [T],
        rank: i32,
        tag: i32,
    ) -> Result<Promise<'a, '_, T>, MpiError> {
        debug_objs!("Communicator", "Send data to {rank} with tag {tag}");
        let req = isend(buf, rank, tag, self.comm_id)? as *mut Request;

        Ok(Promise::new(unsafe { &mut *req }))
    }

    pub fn recv_slice<'a, T: Typed>(
        &self,
        buf: &'a mut [T],
        rank: i32,
        tag: i32,
    ) -> Result<Promise<'a, '_, T>, MpiError> {
        debug_objs!("Communicator", "Recover data from {rank} with tag {tag}");
        let req = irecv(buf, rank, tag, self.comm_id)? as *mut Request;

        Ok(Promise::new(unsafe { &mut *req }))
    }

    pub fn send<'a, T: Typed, A: Deref<Target = [T]>>(
        &self,
        buff: &'a A,
        rank: i32,
        tag: i32,
    ) -> Result<Promise<'a, '_, T>, MpiError> {
        self.send_slice(buff.deref(), rank, tag)
    }

    pub fn recv<'a, T: Typed, A: DerefMut<Target = [T]>>(
        &self,
        buff: &'a mut A,
        rank: i32,
        tag: i32,
    ) -> Result<Promise<'a, '_, T>, MpiError> {
        self.recv_slice(buff.deref_mut(), rank, tag)
    }

    pub fn send_str<'a>(
        &self,
        buff: &'a str,
        rank: i32,
        tag: i32,
    ) -> Result<Promise<'a, '_, u8>, MpiError> {
        self.send_slice(buff.as_bytes(), rank, tag)
    }
}
