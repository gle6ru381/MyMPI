use std::{
    alloc::{alloc, dealloc, Layout},
    slice::from_raw_parts_mut,
};

use libc::c_void;

use mpi::memory::*;

macro_rules! test_cpy {
    ($name:ident, $func:ident) => {
        #[test]
        fn $name() {
            for size in [
                15, 16, 17, 19, 3, 6, 20, 32, 63, 89, 105, 500, 512, 1024, 1500, 2123,
            ] {
                let layout = Layout::from_size_align(size, 32).unwrap();
                unsafe {
                    let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
                    for (i, val) in a.iter_mut().enumerate() {
                        *val = i as i32;
                    }
                    let b = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
                    $func(
                        b.as_mut_ptr() as *mut c_void,
                        a.as_ptr() as *const c_void,
                        size,
                    );
                    for (i, val) in b.into_iter().enumerate() {
                        assert_eq!(*val, i as i32, "Size: {}", size);
                    }
                    dealloc(a.as_mut_ptr() as *mut u8, layout);
                    dealloc(b.as_mut_ptr() as *mut u8, layout);
                }
            }
        }
    };
}

macro_rules! test_aligned {
    ($name:ident, $func:ident) => {
        #[test]
        fn $name() {
            for size in [256, 512, 1024, 2048, 4096] {
                let layout = Layout::from_size_align(size, 32).unwrap();
                unsafe {
                    let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
                    for (i, val) in a.iter_mut().enumerate() {
                        *val = i as i32;
                    }
                    let b = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
                    $func(
                        b.as_mut_ptr() as *mut c_void,
                        a.as_ptr() as *const c_void,
                        size,
                    );
                    for (i, val) in b.into_iter().enumerate() {
                        assert_eq!(*val, i as i32, "Size: {}", size);
                    }
                    dealloc(a.as_mut_ptr() as *mut u8, layout);
                    dealloc(b.as_mut_ptr() as *mut u8, layout);
                }
            }
        }
    };
}

test_cpy!(ymmntcpy_test, ymmntcpy);
test_aligned!(ymmntcpy_aligned_test, ymmntcpy_aligned);
test_cpy!(ymmntcpy_short_test, ymmntcpy_short);
test_cpy!(ymmntcpy_short_prefetch_test, ymmntcpy_short_prefetch);
test_aligned!(
    ymmntcpy_short_prefetch_alignment_test,
    ymmntcpy_short_prefetch_aligned
);
