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
    //let size = MpiObject::size();
    let rank = MpiObject::rank();

    if rank == 0 {
        obj.send_raw(b"Hello world!!!\0", 1, 0, MPI_COMM_WORLD);
        obj.send(&vec![1, 2, 3, 4, 5], 1, 1, MPI_COMM_WORLD);
        let mut data: Data<i32> = obj.recv(5, 1, 10, MPI_COMM_WORLD).unwrap();
        assert_eq!(data.into_slice(), [5, 4, 3, 2, 1]);
    } else {
        let data: Data<u8> = obj.recv(15, 0, 0, MPI_COMM_WORLD).unwrap();
        let str = unsafe { CStr::from_ptr(data.raw() as *const i8).to_str().unwrap() };
        assert_eq!(str, "Hello world!!!");
        let mut data: Data<i32> = obj.recv(5, 0, 1, MPI_COMM_WORLD).unwrap();
        assert_eq!(data.into_slice(), [1, 2, 3, 4, 5]);
        obj.send(&vec![5, 4, 3, 2, 1], 0, 10, MPI_COMM_WORLD);
    }
}

#[test]
fn test_obj_async() {
    set_var("MPI_SIZE", "2");

    let mut obj = MpiObject::new();
    let rank = MpiObject::rank();

    if rank == 0 {
        let mut req = obj
            .send_req_str("First String", 1, 5, MPI_COMM_WORLD)
            .unwrap();
        let mut req2 = obj
            .send_req_raw(&[100, 52141, 7765, -1241], 1, 0, MPI_COMM_WORLD)
            .unwrap();
        obj.wait_all(&mut [req.request(), req2.request()]);
        let mut req = obj.recv_req::<u8>(15, 1, 0, MPI_COMM_WORLD).unwrap();
        let mut req2 = obj.recv_req::<i32>(8, 1, 5, MPI_COMM_WORLD).unwrap();
        let val = String::from_utf8_lossy(req.data());
        assert_eq!(val, "Hello world!!!!");
        assert_eq!(
            req2.data(),
            [-110, 0, 2412, 66654, 41241, 586764, -24124, 4241]
        );
    } else {
        let mut req = obj.recv_req::<i32>(4, 0, 0, MPI_COMM_WORLD).unwrap();
        let mut req2 = obj.recv_str(12, 0, 5, MPI_COMM_WORLD).unwrap();
        assert_eq!(req.data(), [100, 52141, 7765, -1241]);
        let val = String::from_utf8_lossy(req2.into_slice());
        assert_eq!(val, "First String");
        let mut req = obj
            .send_req_raw(
                &[-110, 0, 2412, 66654, 41241, 586764, -24124, 4241],
                0,
                5,
                MPI_COMM_WORLD,
            )
            .unwrap();
        let mut req2 = obj
            .send_req_str("Hello world!!!!", 0, 0, MPI_COMM_WORLD)
            .unwrap();
        obj.wait_all(&mut [req.request(), req2.request()]);
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

    MPI_Bcast(
        rbuf as *mut c_void,
        expect.len() as i32,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD,
    );

    assert_eq!(unsafe { from_raw_parts(rbuf, expect.len()) }, expect);

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

    MPI_Finalize();
}
