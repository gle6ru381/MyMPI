use std::alloc::Layout;
use std::slice::from_raw_parts_mut;
use std::time::Instant;
use std::{ffi::CStr, mem::MaybeUninit, time::Duration, str::Bytes};
use std::alloc;

use mpi::*;
use zstr::zstr;

fn main() {
    let mut tmp = 3;
    let argc = &mut tmp;
    let mut argv = (&mut[zstr!("my_mpi").as_ptr(), zstr!("-n").as_ptr(), zstr!("2").as_ptr()]).as_mut_ptr() as *mut*mut i8;

    std::env::set_var("MPI_SIZE", "4");

    let size = 1024 * 1024 * 100;

    let layout = Layout::from_size_align(size * 4, 32).unwrap();
    let a = unsafe {from_raw_parts_mut(alloc::alloc(layout) as *mut i32, size)};
    for (i, val) in a.iter_mut().enumerate() {
        *val = (i * 2) as i32;
    }
    let b = unsafe {from_raw_parts_mut(alloc::alloc(layout) as *mut i32, size)};

    let now = Instant::now();
    mpi::memory::ymmntcpy(b.as_mut_ptr() as *mut i32 as *mut c_void, a.as_ptr() as *const c_void, 4 * size);
    let val = now.elapsed().as_micros();
    println!("Elapsed time: {val}");

    for i in 0..size {
        if a[i] != b[i] {
            println!("Wrong copy!!");
            break;
        }
    }

    for (i, val) in a.iter_mut().enumerate() {
        *val = (i * 4) as i32;
    }

    let now = Instant::now();
    unsafe {
        b.as_mut_ptr().copy_from(a.as_ptr(),  (size) as usize);
    }
    let val = now.elapsed().as_micros();
    println!("Default elapsed: {val}");
    // println!("MPI init");

    // MPI_Init(argc, &mut argv);

    // let mut size : i32 = 0;
    // let mut rank : i32 = 0;

    // MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    // println!("\n!!!!!!!!!!!!!\nComm size: {size}, rank:{rank}\n!!!!!!!!!!!!!\n");

    // MPI_Barrier(MPI_COMM_WORLD);

    // println!("*************Barrier for {rank}*************");

    // let layout = std::alloc::Layout::from_size_align(100, 1).unwrap();
    // let mut reduceVal = rank;

    // if rank == 0 {
    //     let buff = b"Hello world!!!\0";
    //     unsafe {
    //         let sbuf = std::alloc::alloc(layout);
    //         let unexp = b"##Unexpected message for rank 5.\0";
    //         let bcastmes = b"Bcast message!!!.\0";
    //         let firstBcast = b"First bcast meesage!!!.\0";
    //         let mut reqs : [MPI_Request; 2] = MaybeUninit::uninit().assume_init();
    //         let mut stats : [MPI_Status; 2] = MaybeUninit::uninit().assume_init();

    //         sbuf.copy_from(firstBcast.as_ptr(), firstBcast.len());
    //         MPI_Bcast(sbuf as *mut c_void, firstBcast.len() as i32, MPI_BYTE, 0, MPI_COMM_WORLD);

    //         let mut reduceResult : i32 = uninit!();

    //         MPI_Reduce(&reduceVal as *const i32 as *const c_void, &mut reduceResult as *mut i32 as *mut c_void, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    //         println!("@@@@@@@@Reduce result: {reduceResult}");

    //         let buf = "Part 1";

    //       //  sbuf.copy_from(buff.as_ptr(), 15);
    //       //  MPI_Isend(sbuf as *const c_void, buff.len() as i32, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &mut reqs[0]);
    //       //  MPI_Isend(unexp.as_ptr() as *const c_void, unexp.len() as i32, MPI_BYTE, 1, 5, MPI_COMM_WORLD, &mut reqs[1]);
    //       //  MPI_Waitall(2, reqs.as_mut_ptr(), stats.as_mut_ptr());
    //       //  sbuf.copy_from(bcastmes.as_ptr(), bcastmes.len());
    //       //  MPI_Bcast(sbuf as *mut c_void, bcastmes.len() as i32, MPI_BYTE, 0, MPI_COMM_WORLD);
    //       //  println!("!!!Send end");
    //         std::thread::sleep(Duration::from_secs(1));
    //     }
    // } else {
    //     unsafe {
    //         let buff = std::alloc::alloc(layout);
    //         let unexp = std::alloc::alloc(layout);
    //         let mut stats : [MPI_Status; 2] = MaybeUninit::uninit().assume_init();
    //         let mut reqs : [MPI_Request; 2] = MaybeUninit::uninit().assume_init();

    //         MPI_Bcast(buff as *mut c_void, 24, MPI_BYTE, 0, MPI_COMM_WORLD);
    //         let f = CStr::from_ptr(buff as *const i8).to_str().unwrap();
    //         println!("***First bcast data: {f} from {}", rank);

    //         MPI_Reduce(&reduceVal as *const i32 as *const c_void, &mut reduceVal as *mut i32 as *mut c_void, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    //      //   MPI_Irecv(unexp as *mut c_void, 34, MPI_BYTE, 0, 5, MPI_COMM_WORLD, &mut reqs[0]);
    //      //   MPI_Irecv(buff as *mut c_void, 15, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &mut reqs[1]);

    //         println!("!!!!!!!!Recvover end");

    //      //   MPI_Waitall(2, reqs.as_mut_ptr(), stats.as_mut_ptr());

    //         let data = CStr::from_ptr(buff as *const i8).to_str().unwrap();
    //     //    println!("Unexpt data: {}", CStr::from_ptr(unexp as *const i8).to_str().is_ok());
    //         let udata = CStr::from_ptr(unexp as *const i8).to_str().unwrap();
    //     //    println!("Data: {data}");
    //     //    println!("Unexpected data: {udata}");
    //      //   MPI_Bcast(buff as *mut c_void, 18, MPI_BYTE, 0, MPI_COMM_WORLD);
    //         let bdata = CStr::from_ptr(buff as *const i8).to_str().unwrap();
    //      //   println!("BCast data: {bdata}");
    //     }
    // }
    // println!("Process {rank} exit");
}