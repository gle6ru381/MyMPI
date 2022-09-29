#![allow(non_camel_case_types, non_snake_case)]

mod base;
mod collectives;
mod comm;
mod context;
mod debug;
mod errhandle;
pub mod memory;
mod metatypes;
mod private;
mod reducefuc;
mod reqqueue;
mod shm;
mod types;
mod xfer;

pub use base::*;
pub use collectives::*;
pub use comm::*;
pub use errhandle::*;
pub use metatypes::*;
pub use types::*;
pub use xfer::*;

#[cfg(test)]
mod tests {
    use std::{
        alloc::{alloc, dealloc, Layout},
        slice::from_raw_parts_mut,
        time::{Duration, Instant},
    };

    use libc::c_void;

    use crate::{
        memory::{xmmntcpy, ymmntcpy, ymmntcpy_prefetch, ymmntcpy_short, ymmntcpy_short_prefetch},
        p_mpi_rank_unmap,
    };

    fn createArray() -> Vec<i32> {
        return Vec::with_capacity(983040);
    }

    #[inline(never)]
    fn procWithArray(vec: &Vec<i32>) {
        for i in 0..vec.len() {
            assert_eq!(i as i32, vec[i]);
        }
    }

    #[inline(never)]
    fn fillArray(vec: &mut Vec<i32>) {
        for i in 0..vec.capacity() {
            vec.push(i as i32);
        }
    }

    fn test_xmm(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 16);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            xmmntcpy(
                b.as_mut_ptr() as *mut c_void,
                a.as_mut_ptr() as *const c_void,
                size * 4,
            );
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            // println!("Xmm elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    fn test_avx(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 32);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            ymmntcpy(
                b.as_mut_ptr() as *mut c_void,
                a.as_ptr() as *const c_void,
                size * 4,
            );
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            //            println!("AVX elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    fn test_default(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 4);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            b.as_mut_ptr().copy_from(a.as_ptr(), size);
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            // println!("Default elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    #[inline(never)]
    fn test_default_aligned(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 64);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            //fillArray(&mut vec);

            //let now = Instant::now();
            let tb = b.as_mut_ptr();
            let ta = a.as_mut_ptr();
            tb.copy_from(ta, size);
            //let time = now.elapsed().as_micros();
            let time = 0;
            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            //println!("Default aligned elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    fn test_avx_prefetch(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 32);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            ymmntcpy_prefetch(
                b.as_mut_ptr() as *mut c_void,
                a.as_ptr() as *const c_void,
                size * 4,
            );
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            //            println!("AVX elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    fn test_avx_short(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 32);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            ymmntcpy_short(
                b.as_mut_ptr() as *mut c_void,
                a.as_ptr() as *const c_void,
                size * 4,
            );
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            //            println!("AVX elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    fn test_avx_short_prefetch(size: usize) -> (u128, u128) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size * 4, 32);
            let mut vec = createArray();
            let a = from_raw_parts_mut(alloc(layout) as *mut i32, size);
            for (i, val) in a.iter_mut().enumerate() {
                *val = (i * 4) as i32;
            }
            let b = from_raw_parts_mut(alloc(layout) as *mut i32, size);

            fillArray(&mut vec);

            let now = Instant::now();
            ymmntcpy_short_prefetch(
                b.as_mut_ptr() as *mut c_void,
                a.as_ptr() as *const c_void,
                size * 4,
            );
            let time = now.elapsed().as_micros();

            let now = Instant::now();
            procWithArray(&vec);
            let arr_time = now.elapsed().as_micros();

            //            println!("AVX elapsed: {time}, process: {arr_time}");

            for i in 0..size {
                assert_eq!(a[i], b[i]);
            }

            dealloc(a.as_mut_ptr() as *mut u8, layout);
            dealloc(b.as_mut_ptr() as *mut u8, layout);

            (time, arr_time)
        }
    }

    const LOOPS: u128 = 100;

    #[test]
    fn full_test() {
        let mut time = (0u128, 0u128);
        for size in [
            4096, 16384, 65536, 262144, 1048576, 1310720, 4194304, 8388608, 9699328,
        ] {
            // println!("Size: {size}");
            // time = (0u128, 0u128);
            // for _ in 0..LOOPS {
            //     let val = test_avx(size);
            //     time.0 += val.0;
            //     time.1 += val.1;
            // }
            // println!(
            //     "AVX elapsed time: {}, after process time: {}",
            //     time.0 / LOOPS,
            //     time.1 / LOOPS
            // );

            // time = (0, 0);
            // for _ in 0..LOOPS {
            //     let val = test_xmm(size);
            //     time.0 += val.0;
            //     time.1 += val.1;
            // }
            // println!(
            //     "Xmm elapsed time: {}, after process time: {}",
            //     time.0 / LOOPS,
            //     time.1 / LOOPS
            // );

            // time = (0, 0);
            // for _ in 0..LOOPS {
            //     let val = test_default(size);
            //     time.0 += val.0;
            //     time.1 += val.1;
            // }
            // println!(
            //     "Default elapsed time: {}, after process time: {}",
            //     time.0 / LOOPS,
            //     time.1 / LOOPS
            // );

            // time = (0, 0);
            for _ in 0..LOOPS {
                let val = test_default_aligned(size);
                time.0 += val.0;
                time.1 += val.1;
            }
            println!(
                "Default aligned elapsed time: {}, after process time: {}",
                time.0 / LOOPS,
                time.1 / LOOPS
            );

            time = (0, 0);
            for _ in 0..LOOPS {
                let val = test_avx_prefetch(size);
                time.0 += val.0;
                time.1 += val.1;
            }
            println!(
                "AVX prefetch elapsed time: {}, after process time: {}",
                time.0 / LOOPS,
                time.1 / LOOPS
            );

            time = (0, 0);
            for _ in 0..LOOPS {
                let val = test_avx_short(size);
                time.0 += val.0;
                time.1 += val.1;
            }
            println!(
                "AVX short elapsed time: {}, after process time: {}",
                time.0 / LOOPS,
                time.1 / LOOPS
            );

            time = (0, 0);
            for _ in 0..LOOPS {
                let val = test_avx_short_prefetch(size);
                time.0 += val.0;
                time.1 += val.1;
            }
            println!(
                "AVX short prefetch elapsed time: {}, after process time: {}",
                time.0 / LOOPS,
                time.1 / LOOPS
            );
        }
    }
}
