use std::{ffi::CStr};

use mpi::*;
use zstr::zstr;

fn main() {
    let mut tmp = 3;
    let argc = &mut tmp;
    let mut argv = (&mut[zstr!("my_mpi").as_ptr(), zstr!("-n").as_ptr(), zstr!("2").as_ptr()]).as_mut_ptr() as *mut*mut i8;

    std::env::set_var("MPI_SIZE", "2");

    println!("MPI init");

    MPI_Init(argc, &mut argv);

    let mut size : i32 = 0;
    let mut rank : i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    println!("Comm size: {size}, rank:{rank}");

    let layout = std::alloc::Layout::from_size_align(100, 1).unwrap();

    if rank == 0 {
        let buff = b"Hello world!!!\0";
        unsafe {
            let sbuf = std::alloc::alloc(layout);
            sbuf.copy_from(buff.as_ptr(), 15);
            MPI_Send(sbuf as *mut c_void, buff.len() as i32, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        }
    } else {
        unsafe {
            let buff = std::alloc::alloc(layout);
            let mut status = MPI_Status{MPI_SOURCE: 0, MPI_ERROR: 0, MPI_TAG: 0, cnt: 0};
            MPI_Recv(buff as *mut c_void, 15, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mut status);

            let data = CStr::from_ptr(buff as *const i8).to_str().unwrap();
            println!("Data: {data}");
        }
    }
}