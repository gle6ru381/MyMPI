use std::{
    alloc::{alloc, dealloc, Layout},
    slice::{from_raw_parts_mut}
};

use libc::c_void;

use mpi::{
    memory::*
};

macro_rules! test_cpy {
    ($name:ident, $func:ident, $size:literal) => {
        #[test]
        fn $name() {
            let layout = Layout::from_size_align($size, 32).unwrap();
            unsafe {
                let a = from_raw_parts_mut(alloc(layout) as *mut i32, $size / 4);
                for (i, val) in a.iter_mut().enumerate() {
                    *val = i as i32;
                }
                let b = from_raw_parts_mut(alloc(layout) as *mut i32, $size / 4);
                $func(b.as_mut_ptr() as *mut c_void, a.as_ptr() as *const c_void, $size);
                for (i, val) in b.into_iter().enumerate() {
                    assert_eq!(*val, i as i32)
                }
                dealloc(a.as_mut_ptr() as *mut u8, layout);
                dealloc(b.as_mut_ptr() as *mut u8, layout);
            }
        }
    };
}

test_cpy!(ymmntcpy_test, ymmntcpy, 1024);
test_cpy!(ymmntcpy_aligned_test, ymmntcpy_aligned, 1024);
test_cpy!(ymmntcpy_short_test, ymmntcpy_short, 1024);
test_cpy!(ymmntcpy_short_prefetch_test, ymmntcpy_short_prefetch, 1024);
test_cpy!(ymmntcpy_short_prefetch_alignment_test, ymmntcpy_short_prefetch_aligned, 1024);