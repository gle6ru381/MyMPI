use crate::{MPI_Comm, MPI_Status, MpiResult, MPI_Request, uninit, MpiError};

use self::{send::isend, recv::irecv};

use super::request::Request;

pub(crate) mod recv;
pub(crate) mod send;

pub fn sendrecv(
    sbuf: &[u8],
    dest: i32,
    stag: i32,
    rbuf: &mut [u8],
    src: i32,
    rtag: i32,
    comm: MPI_Comm,
) -> Result<MPI_Status, MpiError> {

    let mut req: [*mut Request; 2] = uninit();
    let mut stat: [MPI_Status; 2] = uninit();

    isend(sbuf, dest, stag, comm, unsafe {&mut&mut *req[0]})?;
    irecv(rbuf, src, rtag, comm, unsafe {&mut&mut *req[1]})?;
    Request::wait_all(&mut req, &mut stat)?;

    Ok(stat[1])
}
