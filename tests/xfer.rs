use mpi::*;
use std::{
    alloc::{alloc, dealloc, Layout},
    env::set_var,
    ffi::CStr,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[test]
fn test_p2p() {
    set_var("MPI_SIZE", "2");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let layout = Layout::from_size_align(100, 32).unwrap();

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
            debug_assert!(buff as usize % 32 == 0);
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

    let layout = Layout::from_size_align(100, 32).unwrap();
    if rank == 0 {
        let buff = b"Hello world!!!\0";
        let unexpect = b"Unexpect message\0";
        let rbuf = unsafe { alloc(layout) };
        debug_assert!(rbuf as usize % 32 == 0);
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
            debug_assert!(buff as usize % 32 == 0);
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

#[test]
fn test_obj() {
    set_var("MPI_SIZE", "2");

    let mut obj = MpiObject::new();
    let rank = MpiObject::rank();
    let comm = obj.get_comm(MPI_COMM_WORLD).unwrap();

    if rank == 0 {
        comm.send_slice(b"Hello world!!!\0", 1, 0).unwrap();
        comm.send(&vec![1, 2, 3, 4, 5], 1, 1).unwrap();
        let mut data: Data<i32> = Data::new(5);
        comm.recv(&mut data, 1, 10).unwrap();
        assert_eq!(data.into_slice(), [5, 4, 3, 2, 1]);
    } else {
        let mut data: Data<u8> = Data::new(15);
        comm.recv(&mut data, 0, 0).unwrap();
        let str = unsafe { CStr::from_ptr(data.raw() as *const i8).to_str().unwrap() };
        assert_eq!(str, "Hello world!!!");
        let mut arr: Data<i32> = Data::new(5);
        comm.recv(&mut arr, 0, 1).unwrap();
        assert_eq!(arr.into_slice(), [1, 2, 3, 4, 5]);
        comm.send(&vec![5, 4, 3, 2, 1], 0, 10).unwrap();
    }
}

#[test]
fn test_obj_async() {
    set_var("MPI_SIZE", "2");

    let mut obj = MpiObject::new();
    let rank = MpiObject::rank();
    let comm = obj.get_comm(MPI_COMM_WORLD).unwrap();

    if rank == 0 {
        let p1 = comm.send_str("First String", 1, 5).unwrap();
        let p2 = comm.send_slice(&[100, 52141, 7765, -1241], 1, 0).unwrap();
        p1.wait().unwrap();
        p2.wait().unwrap();

        let mut d2: Data<i32> = Data::new(8);
        let p4 = comm.recv(&mut d2, 1, 5).unwrap();
        let mut d: Data<u8> = Data::new(14);
        let p3 = comm.recv(&mut d, 1, 1).unwrap();

        p4.wait().unwrap();
        p3.wait().unwrap();

        assert_eq!(to_string!(d), "Hello world!!!");

        assert_eq!(
            d2.into_slice(),
            [-110, 0, 2412, 66654, 41241, 586764, -24124, 4241]
        );
    } else {
        let mut d: Data<i32> = Data::new(4);
        let p1 = comm.recv(&mut d, 0, 0).unwrap();
        let mut d2: Data<u8> = Data::new(12);
        let p2 = comm.recv(&mut d2, 0, 5).unwrap();

        p1.wait().unwrap();
        p2.wait().unwrap();

        assert_eq!(d.into_slice_mut(), [100, 52141, 7765, -1241]);
        let val = String::from_utf8_lossy(d2.into_slice());
        assert_eq!(val, "First String");

        let p3 = comm
            .send_slice(&[-110, 0, 2412, 66654, 41241, 586764, -24124, 4241], 0, 5)
            .unwrap();
        let p4 = comm.send_str("Hello world!!!", 0, 1).unwrap();

        p4.wait().unwrap();
        p3.wait().unwrap();
    }
}

#[test]
fn test_big_data() {
    set_var("MPI_SIZE", "2");

    MPI_Init(null_mut(), null_mut());
    let mut rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let msg_sizes = [4096, 102400, 204800, 409600, 921600, 3145728];

    if rank == 0 {
        for size in msg_sizes.into_iter() {
            let layout = Layout::from_size_align(size as usize, 32).unwrap();
            let vec = unsafe { from_raw_parts_mut(alloc(layout), size as usize) };

            for i in 0..size {
                vec[i as usize] = (i as u8 * size as u8) as u8;
            }
            MPI_Send(
                vec.as_ptr() as *const c_void,
                size,
                MPI_BYTE,
                1,
                0,
                MPI_COMM_WORLD,
            );
            unsafe {
                dealloc(vec.as_mut_ptr(), layout);
            }
        }
    } else {
        for size in msg_sizes.into_iter() {
            let layout = Layout::from_size_align(size as usize, 32).unwrap();
            let vec = unsafe { from_raw_parts_mut(alloc(layout), size as usize) };
            for i in 0..size {
                vec[i as usize] = 0;
            }
            let mut stat = MPI_Status::uninit();
            MPI_Recv(
                vec.as_mut_ptr() as *mut c_void,
                size,
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD,
                &mut stat,
            );
            for i in 0..size {
                assert_eq!(vec[i as usize], (i as u8 * size as u8) as u8);
            }
            unsafe {
                dealloc(vec.as_mut_ptr(), layout);
            }
        }
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

#[test]
fn test_bcast_4() {
    set_var("MPI_SIZE", "4");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let layout = Layout::from_size_align(100, 32).unwrap();

    let rbuf: *mut u8 = unsafe { alloc(layout) };
    let expect = b"Hello world!!!\0";

    if rank == 0 {
        unsafe {
            rbuf.copy_from(expect.as_ptr(), expect.len());
        }
    }

    for _ in 0..100000 {
        MPI_Bcast(
            rbuf as *mut c_void,
            expect.len() as i32,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD,
        );
    }

    assert_eq!(unsafe { from_raw_parts(rbuf, expect.len()) }, expect);

    unsafe {
        dealloc(rbuf, layout);
    }

    MPI_Finalize();
}

#[test]
fn test_bcast_8() {
    set_var("MPI_SIZE", "8");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let layout = Layout::from_size_align(100, 32).unwrap();

    let rbuf: *mut u8 = unsafe { alloc(layout) };
    let expect = b"Hello world!!!\0";

    if rank == 0 {
        unsafe {
            rbuf.copy_from(expect.as_ptr(), expect.len());
        }
    }

    MPI_Bcast(
        rbuf as *mut c_void,
        expect.len() as i32,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD,
    );

    assert_eq!(unsafe { from_raw_parts(rbuf, expect.len()) }, expect);

    unsafe {
        dealloc(rbuf, layout);
    }

    MPI_Finalize();
}

#[test]
fn test_gather_4() {
    set_var("MPI_SIZE", "4");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let mut buf: [u8; 400] = [0; 400];

    for val in (rank * 100)..(rank * 100 + 100) {
        buf[val as usize] = (val + 1) as u8;
    }

    unsafe {
        MPI_Gather(
            buf.as_ptr().add((rank * 100) as usize) as *mut c_void,
            100,
            MPI_BYTE,
            buf.as_mut_ptr() as *mut c_void,
            100,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD,
        );
    }

    if rank == 0 {
        for val in 0..400 {
            assert_eq!(buf[val as usize], (val + 1) as u8);
        }
    }

    unsafe {
        MPI_Allgather(
            buf.as_ptr().add((rank * 100) as usize) as *mut c_void,
            100,
            MPI_BYTE,
            buf.as_mut_ptr() as *mut c_void,
            100,
            MPI_BYTE,
            MPI_COMM_WORLD,
        );
    }

    for val in 0..400 {
        assert_eq!(buf[val as usize], (val + 1) as u8);
    }

    MPI_Finalize();
}

#[test]
fn test_gather_8() {
    set_var("MPI_SIZE", "8");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let mut buf: [u8; 800] = [0; 800];

    for val in (rank * 100)..(rank * 100 + 100) {
        buf[val as usize] = (val + 1) as u8;
    }

    unsafe {
        MPI_Gather(
            buf.as_ptr().add((rank * 100) as usize) as *mut c_void,
            100,
            MPI_BYTE,
            buf.as_mut_ptr() as *mut c_void,
            100,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD,
        );
    }

    if rank == 0 {
        for val in 0..800 {
            assert_eq!(buf[val as usize], (val + 1) as u8);
        }
    }

    unsafe {
        MPI_Allgather(
            buf.as_ptr().add((rank * 100) as usize) as *mut c_void,
            100,
            MPI_BYTE,
            buf.as_mut_ptr() as *mut c_void,
            100,
            MPI_BYTE,
            MPI_COMM_WORLD,
        );
    }

    for val in 0..800 {
        assert_eq!(buf[val as usize], (val + 1) as u8);
    }

    MPI_Finalize();
}

#[test]
fn test_reduce_4() {
    set_var("MPI_SIZE", "4");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let mut buf: [i32; 400] = [0; 400];
    let mut rbuf: [i32; 100] = [0; 100];

    for val in (rank * 100)..(rank * 100 + 100) {
        buf[val as usize] = val;
    }

    unsafe {
        MPI_Reduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_MAX,
            0,
            MPI_COMM_WORLD,
        );

        if rank == 0 {
            for val in 0..100 {
                assert_eq!(rbuf[val as usize], (size - 1) * 100 + val);
            }
        }

        MPI_Reduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD,
        );

        if rank == 0 {
            for val in 0..100 {
                let mut test = val;
                for v in 1..size {
                    test += v * 100 + val;
                }
                assert_eq!(rbuf[val as usize], test);
            }
        }

        MPI_Allreduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD,
        );

        for val in 0..100 {
            assert_eq!(rbuf[val as usize], val);
        }
    }
    MPI_Finalize();
}

#[test]
fn test_reduce_8() {
    set_var("MPI_SIZE", "8");

    MPI_Init(null_mut(), null_mut());

    let mut size: i32 = 0;
    let mut rank: i32 = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &mut size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mut rank);

    let mut buf: [i32; 800] = [0; 800];
    let mut rbuf: [i32; 100] = [0; 100];

    for val in (rank * 100)..(rank * 100 + 100) {
        buf[val as usize] = val;
    }

    unsafe {
        MPI_Reduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_MAX,
            0,
            MPI_COMM_WORLD,
        );

        if rank == 0 {
            for val in 0..100 {
                assert_eq!(rbuf[val as usize], (size - 1) * 100 + val);
            }
        }

        MPI_Reduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_SUM,
            0,
            MPI_COMM_WORLD,
        );

        if rank == 0 {
            for val in 0..100 {
                let mut test = val;
                for v in 1..size {
                    test += v * 100 + val;
                }
                assert_eq!(rbuf[val as usize], test);
            }
        }

        MPI_Allreduce(
            buf.as_mut_ptr().add((rank * 100) as usize) as *mut c_void,
            rbuf.as_mut_ptr() as *mut c_void,
            100,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD,
        );

        for val in 0..100 {
            assert_eq!(rbuf[val as usize], val);
        }
    }
    MPI_Finalize();
}
