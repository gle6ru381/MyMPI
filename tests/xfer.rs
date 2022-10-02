use mpi::*;
use std::{
    alloc::{alloc, dealloc, Layout},
    env::set_var,
    ffi::CStr,
    ptr::{null, null_mut},
};

#[test]
fn test_p2p() {
    set_var("MPI_SIZE", "2");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let layout = Layout::from_size_align(100, 1).unwrap();

    if rank == 0 {
        let buff = b"Hello world!!!\0";
        let rbuf = unsafe { alloc(layout) };
        MPI_Send(
            buff.as_ptr() as *const c_void,
            buff.len() as i32,
            MPI_BYTE,
            1,
            0,
            MPI_COMM_WORLD,
        );
        let mut stat = MPI_Status::uninit();
        MPI_Recv(
            rbuf as *mut c_void,
            15,
            MPI_BYTE,
            1,
            1,
            MPI_COMM_WORLD,
            &mut stat,
        );
        let data = unsafe { CStr::from_ptr(rbuf as *const i8).to_str().unwrap() };
        assert_eq!(data, "Hello world!!!");
        unsafe { dealloc(rbuf, layout) };
    } else {
        unsafe {
            let buff = alloc(layout);
            let mut stat = MPI_Status::uninit();
            MPI_Recv(
                buff as *mut c_void,
                15,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD,
                &mut stat,
            );
            let data = CStr::from_ptr(buff as *const i8).to_str().unwrap();
            assert_eq!(data, "Hello world!!!");
            MPI_Send(buff as *const c_void, 15, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            dealloc(buff, layout);
        }
    }
    MPI_Finalize();
}

#[test]
fn test_p2p_unexpect() {
    set_var("MPI_SIZE", "2");

    MPI_Init(null_mut(), null_mut());
    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let layout = Layout::from_size_align(100, 1).unwrap();
    if rank == 0 {
        let buff = b"Hello world!!!\0";
        let unexpect = b"Unexpect message\0";
        let rbuf = unsafe { alloc(layout) };

        let mut reqs: [MPI_Request; 2] = uninit();
        let mut stats: [MPI_Status; 2] = uninit();
        MPI_Isend(
            unexpect.as_ptr() as *const c_void,
            unexpect.len() as i32,
            MPI_BYTE,
            1,
            1,
            MPI_COMM_WORLD,
            &mut reqs[0],
        );
        MPI_Isend(
            buff.as_ptr() as *const c_void,
            buff.len() as i32,
            MPI_BYTE,
            1,
            0,
            MPI_COMM_WORLD,
            &mut reqs[1],
        );
        MPI_Waitall(2, reqs.as_mut_ptr(), stats.as_mut_ptr());
        MPI_Recv(
            rbuf as *mut c_void,
            17,
            MPI_BYTE,
            1,
            0,
            MPI_COMM_WORLD,
            &mut stats[0],
        );
        let data = unsafe { CStr::from_ptr(rbuf as *const i8).to_str().unwrap() };
        assert_eq!(data, "Unexpect message");
        unsafe { dealloc(rbuf, layout) };
    } else {
        unsafe {
            let buff = alloc(layout);
            let mut stat = MPI_Status::uninit();
            MPI_Recv(
                buff as *mut c_void,
                15,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD,
                &mut stat,
            );
            let data = CStr::from_ptr(buff as *const i8).to_str().unwrap();
            assert_eq!(data, "Hello world!!!");
            MPI_Recv(
                buff as *mut c_void,
                17,
                MPI_BYTE,
                0,
                1,
                MPI_COMM_WORLD,
                &mut stat,
            );
            let data = CStr::from_ptr(buff as *const i8).to_str().unwrap();
            assert_eq!(data, "Unexpect message");
            MPI_Send(buff as *const c_void, 17, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

// #[test]
// fn test_buff() {
//     set_var("MPI_SIZE", "2");
//     MPI_Init(null_mut(), null_mut());
//     let mut rank = 0;
//     let mut size = 0;

//     MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &mut size);

//     if rank == 0 {
//         let mut buff = Buffer::from(b"Hello world!!!")
//             .set_comm(MPI_COMM_WORLD)
//             .set_rank(1)
//             .set_tag(0);
//         let mut obj = MpiObject::new();
//         obj.send(&buff);
//         buff = buff.set_data(&vec![1, 2, 3, 4, 5]).set_tag(1);
//         obj.send(&buff);
//         buff = buff.set_data_raw("data".as_bytes()).set_tag(2);
//         obj.send(&buff);
//         obj.wait_all(None);
//     } else {
//         let mut buff = Buffer::from_mut(&mut [0u8; 100])
//             .set_comm(MPI_COMM_WORLD)
//             .set_rank(0)
//             .set_tag(0);
//         let mut obj = MpiObject::new();
//         let idx = obj.recv(&buff);
//         obj.wait(idx);
//         let data = unsafe {CStr::from_ptr(buff.data().as_ptr()).to_str().unwrap()};
//         assert_eq!(data, "Hello world!!!");
//         buff = buff.set_tag(1);
//         let idx = obj.recv(&buff);
//         obj.wait(idx);
//         let data: Vec<i32> = Vec::from(buff.data_mut());
//         assert_eq!(data, vec![1, 2, 3, 4, 5]);
//         buff = buff.set_tag(2);
//         let idx = obj.recv(&buff);
//         obj.wait(idx);
//         let data = unsafe {CStr::from_ptr(buff.data().as_ptr()).to_str().unwrap()};
//         assert_eq!(data, "data");
//     }

//     MPI_Finalize();
// }
