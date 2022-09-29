use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mpi::memory::*;
use std::slice::{from_raw_parts_mut, from_raw_parts};
use std::alloc::*;
use std::time::Instant;
use std::ffi::c_void;

fn createArray() -> Vec<i32> {
    let mut vec = Vec::with_capacity(883040);
    for i in 0..vec.capacity() {
        vec.push(i as i32);
    }
    vec
}

#[inline(never)]
fn procWithArray(vec : &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[inline(never)]
fn fillArray(vec : &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[inline(never)]
fn begin_bench() -> Vec<i32> {
    let mut arr = createArray();
    fillArray(&mut arr);
    arr
}

fn allocate_ymm(size : usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        for (i, val) in a.iter_mut().enumerate() {
            *val = (i * 4) as i32;
        }
        let b = alloc(layout) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

fn dealoc_ymm(size : usize, a : *mut c_void, b : *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn allocate_xmm(size : usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 16).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        for (i, val) in a.iter_mut().enumerate() {
            *val = (i * 4) as i32;
        }
        let b = alloc(layout) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

fn dealoc_xmm(size : usize, a : *mut c_void, b : *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 16).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn allocate_default(size : usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 4).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        for (i, val) in a.iter_mut().enumerate() {
            *val = (i * 4) as i32;
        }
        let b = alloc(layout) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

fn dealoc_default(size : usize, a : *mut c_void, b : *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 4).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}


fn cpy_benchmark(c : &mut Criterion) {
    let mut g = c.benchmark_group("Cpy group");
    g.noise_threshold(0.0001);
    g.confidence_level(0.99);
    for size in [4096, 16384, 65536, 262144, 1048576, 1310720, 4194304, 8388608, 9699328] {
        let mut data = allocate_xmm(size);
        let mut vec = begin_bench();

        g.bench_function(String::from("Bench xmm ") + &size.to_string(), |x| {
            x.iter(|| {
                fillArray(&mut vec);
                xmmntcpy(data.1, data.0, size);
                procWithArray(&mut vec);
            })
        });
        dealoc_xmm(size, data.0, data.1);

        data = allocate_ymm(size);
        g.bench_function(String::from("Bench ymm ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            ymmntcpy(data.1, data.0, size);
            procWithArray(&mut vec);
        }));
        dealoc_ymm(size, data.0, data.1);

        data = allocate_default(size);
        g.bench_function(String::from("Bench default ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            unsafe {std::ptr::copy(data.1, data.0, size)};
            procWithArray(&mut vec);
        }));
        dealoc_default(size, data.0, data.1);

        data = allocate_ymm(size);
        g.bench_function(String::from("Bench default align ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            unsafe {std::ptr::copy(data.1, data.0, size)};
            procWithArray(&mut vec);
        }));
        dealoc_ymm(size, data.0, data.1);

        data = allocate_ymm(size);
        g.bench_function(String::from("Bench ymm prefetch ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            ymmntcpy_prefetch(data.1, data.0, size);
            procWithArray(&mut vec);
        }));
        dealoc_ymm(size, data.0, data.1);

        data = allocate_ymm(size);
        g.bench_function(String::from("Bench ymm short ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            ymmntcpy(data.1, data.0, size);
            procWithArray(&mut vec);
        }));
        dealoc_ymm(size, data.0, data.1);

        data = allocate_ymm(size);
        g.bench_function(String::from("Bench ymm short prefetch ") + &size.to_string(), |x| x.iter(|| {
            fillArray(&mut vec);
            ymmntcpy(data.1, data.0, size);
            procWithArray(&mut vec);
        }));
        dealoc_ymm(size, data.0, data.1);
    }
}

criterion_group!(benches, cpy_benchmark);
criterion_main!(benches);