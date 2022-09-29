use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mpi::memory::*;
use std::alloc::*;
use std::ffi::c_void;
use std::slice::{from_raw_parts, from_raw_parts_mut};

fn createArray(size : usize) -> Vec<i32> {
    let mut vec = Vec::with_capacity(size / 4);
    for i in 0..vec.capacity() {
        vec.push(i as i32);
    }
    vec
}

#[inline(never)]
fn procWithArray(vec: &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[inline(never)]
fn fillArray(vec: &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

fn allocate_ymm(size: usize) -> (*mut c_void, *mut c_void) {
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

fn dealoc_ymm(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn allocate_xmm(size: usize) -> (*mut c_void, *mut c_void) {
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

fn dealoc_xmm(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 16).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn allocate_default(size: usize) -> (*mut c_void, *mut c_void) {
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

fn dealoc_default(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 4).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn cpy_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("Cpy group");
    g.noise_threshold(0.0001);
    g.significance_level(0.01);
    g.confidence_level(0.99);
    for vec_size in [4096, 262144, 524288, 1000000, 6291456] {
        for size in [
            4096, 16384, 65536, 262144, 1048576, 1310720, 4194304, 8388608, 9699328,
        ] {
            let mut data = allocate_xmm(size);
            let mut vec = createArray(vec_size);

            g.bench_function(format!("Bench xmm, vector size: {vec_size}, block size: {size}"), |x| {
                x.iter(|| {
                    fillArray(&mut vec);
                    xmmntcpy(data.1, data.0, size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_xmm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_function(format!("Bench ymm, vector size: {vec_size}, block size: {size}"), |x| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy(data.1, data.0, size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            data = allocate_default(size);
            g.bench_function(format!("Bench default, vector size: {vec_size}, block size: {size}"), |x| {
                x.iter(|| {
                    fillArray(&mut vec);
                    unsafe { std::ptr::copy(data.1, data.0, size) };
                    procWithArray(&mut vec);
                })
            });
            dealoc_default(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_function(
                format!("Bench default align, vector size: {vec_size}, block size: {size}"),
                |x| {
                    x.iter(|| {
                        fillArray(&mut vec);
                        unsafe { std::ptr::copy(data.1, data.0, size) };
                        procWithArray(&mut vec);
                    })
                },
            );
            dealoc_ymm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_function(
                format!("Bench ymm prefetch, vector size: {vec_size}, block size: {size}"),
                |x| {
                    x.iter(|| {
                        fillArray(&mut vec);
                        ymmntcpy_prefetch(data.1, data.0, size);
                        procWithArray(&mut vec);
                    })
                },
            );
            dealoc_ymm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_function(format!("Bench ymm short, vector size: {vec_size}, block size: {size}"), |x| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy(data.1, data.0, size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_function(
                format!("Bench ymm short prefetch, vector size: {vec_size}, block size: {size}"),
                |x| {
                    x.iter(|| {
                        fillArray(&mut vec);
                        ymmntcpy(data.1, data.0, size);
                        procWithArray(&mut vec);
                    })
                },
            );
            dealoc_ymm(size, data.0, data.1);
        }
    }
}

criterion_group!(benches, cpy_benchmark);
criterion_main!(benches);
