use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
};
use mpi::memory::*;
use std::alloc::*;
use std::ffi::c_void;
use std::fmt::Display;
use std::ptr::{null, null_mut};
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::time::Duration;

fn createArray(size: usize) -> Vec<i32> {
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
        // for (i, val) in a.iter_mut().enumerate() {
        //     *val = (i * 4) as i32;
        // }
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
        // for (i, val) in a.iter_mut().enumerate() {
        //     *val = (i * 4) as i32;
        // }
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
        let layout = Layout::from_size_align(size + 1, 1).unwrap();
        let a = from_raw_parts_mut(alloc(layout).add(1) as *mut i32, size / 4);
        // for (i, val) in a.iter_mut().enumerate() {
        //     *val = (i * 4) as i32;
        // }
        let b = alloc(layout).add(1) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

fn dealoc_default(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size + 1, 1).unwrap();
        dealloc((a as *mut u8).sub(1), layout);
        dealloc((b as *mut u8).sub(1), layout);
    }
}

fn allocate_avx512(size: usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        let b = alloc(layout) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

fn deallocate_avx512(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 64);
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

fn cpy_benchmark(c: &mut Criterion) {
    for vec_size in [4096, 16384, 131072, 262144, 524288, 786432, 1048576, 4194304, 6291456, 7876608] {
        let mut g = c.benchmark_group(format!("Copy_vec_size_{vec_size}"));
        g.noise_threshold(0.05);
        g.significance_level(0.0001);
        g.confidence_level(0.99999);
        g.warm_up_time(Duration::from_nanos(1));
        g.sampling_mode(criterion::SamplingMode::Linear);
        g.plot_config(
            PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
        );
        g.sample_size(10);
        for size in [4096, 16384, 131072, 262144, 524288, 1048576, 4194304, 6291456, 7876608, 8388608] {
            let mut data = allocate_xmm(size);
            let mut vec = createArray(vec_size);

            // g.throughput(criterion::Throughput::Bytes((vec_size + size) as u64));
            // g.bench_with_input(BenchmarkId::new("128bit", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         xmmntcpy(data.1, data.0, *size);
            //         procWithArray(&mut vec);
            //     })
            // });
            // dealoc_xmm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_with_input(BenchmarkId::new("256bit", size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            data = allocate_default(size);
            g.bench_with_input(BenchmarkId::new("Default", size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    unsafe { std::ptr::copy(data.1, data.0, *size) };
                    procWithArray(&mut vec);
                })
            });
            dealoc_default(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_with_input(BenchmarkId::new("Default_align", size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    unsafe { std::ptr::copy(data.1, data.0, *size) };
                    procWithArray(&mut vec);
                })
            });

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_prefetch", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_prefetch(data.1, data.0, *size);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );

            // g.bench_with_input(BenchmarkId::new("256bit_short", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         ymmntcpy_short(data.1, data.0, *size);
            //         procWithArray(&mut vec);
            //     })
            // });

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_short_prefetch", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_short_prefetch(data.1, data.0, *size);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );

            g.bench_with_input(BenchmarkId::new("256bit_short_prefetch_aligned", size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy_short_prefetch_aligned(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });

            g.bench_with_input(BenchmarkId::new("256bit_prefetch_aligned", size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy_prefetch_aligned(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            // data = allocate_avx512(size);
            // g.bench_function(format!("Bench avx512, vector size: {vec_size}, block size: {size}"), |x| x.iter(|| {
            //     fillArray(&mut vec);
            //     avx512ntcpy(data.1, data.0, size);
            //     procWithArray(&mut vec);
            // }));

            // g.bench_function(format!("Bench avx512 short, vector size: {vec_size}, block size: {size}"), |x| x.iter(|| {
            //     fillArray(&mut vec);
            //     avx512ntcpy_short(data.1, data.0, size);
            //     procWithArray(&mut vec);
            // }));
        }
        g.finish();
    }

    for size in [4096, 16384, 131072, 262144, 524288, 1048576, 4194304, 6291456, 7876608, 8388608] {
        let mut g = c.benchmark_group(format!("Copy_buf_size_{size}"));
        g.noise_threshold(0.05);
        g.significance_level(0.0001);
        g.confidence_level(0.99999);
        g.warm_up_time(Duration::from_nanos(1));
        g.sampling_mode(criterion::SamplingMode::Linear);
        g.plot_config(
            PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
        );
        g.sample_size(5000);
        for vec_size in [4096, 16384, 131072, 262144, 524288, 786432, 1048576, 4194304, 6291456, 7876608] {
            let mut data = allocate_xmm(size);
            let mut vec = createArray(vec_size);

            // g.throughput(criterion::Throughput::Bytes((vec_size + size) as u64));
            // g.bench_with_input(BenchmarkId::new("128bit", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         xmmntcpy(data.1, data.0, *size);
            //         procWithArray(&mut vec);
            //     })
            // });
            // dealoc_xmm(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_with_input(BenchmarkId::new("256bit", vec_size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            data = allocate_default(size);
            g.bench_with_input(BenchmarkId::new("Default", vec_size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    unsafe { std::ptr::copy(data.1, data.0, *size) };
                    procWithArray(&mut vec);
                })
            });
            dealoc_default(size, data.0, data.1);

            data = allocate_ymm(size);
            g.bench_with_input(BenchmarkId::new("Default_align", vec_size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    unsafe { std::ptr::copy(data.1, data.0, *size) };
                    procWithArray(&mut vec);
                })
            });

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_prefetch", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_prefetch(data.1, data.0, *size);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );

            // g.bench_with_input(BenchmarkId::new("256bit_short", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         ymmntcpy_short(data.1, data.0, *size);
            //         procWithArray(&mut vec);
            //     })
            // });

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_short_prefetch", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_short_prefetch(data.1, data.0, *size);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );

            g.bench_with_input(BenchmarkId::new("256bit_short_prefetch_aligned", vec_size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy_short_prefetch_aligned(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });

            g.bench_with_input(BenchmarkId::new("256bit_prefetch_aligned", vec_size), &size, |x, size| {
                x.iter(|| {
                    fillArray(&mut vec);
                    ymmntcpy_prefetch_aligned(data.1, data.0, *size);
                    procWithArray(&mut vec);
                })
            });
            dealoc_ymm(size, data.0, data.1);

            // data = allocate_avx512(size);
            // g.bench_function(format!("Bench avx512, vector size: {vec_size}, block size: {size}"), |x| x.iter(|| {
            //     fillArray(&mut vec);
            //     avx512ntcpy(data.1, data.0, size);
            //     procWithArray(&mut vec);
            // }));

            // g.bench_function(format!("Bench avx512 short, vector size: {vec_size}, block size: {size}"), |x| x.iter(|| {
            //     fillArray(&mut vec);
            //     avx512ntcpy_short(data.1, data.0, size);
            //     procWithArray(&mut vec);
            // }));
        }
        g.finish();
    }
}

// fn fibonacci_slow(n: u64) -> u64 {
//     match n {
//         0 => 1,
//         1 => 1,
//         n => fibonacci_slow(n-1) + fibonacci_slow(n-2),
//     }
// }

// fn fibonacci_fast(n: u64) -> u64 {
//     let mut a = 0;
//     let mut b = 1;

//     match n {
//         0 => b,
//         _ => {
//             for _ in 0..n {
//                 let c = a + b;
//                 a = b;
//                 b = c;
//             }
//             b
//         }
//     }
// }

// fn bench_fibs(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Fibonacci");
//     for i in [20u64, 21u64].iter() {
//         group.bench_with_input(BenchmarkId::new("Recursive", i), i,
//             |b, i| b.iter(|| fibonacci_slow(*i)));
//         group.bench_with_input(BenchmarkId::new("Iterative", i), i,
//             |b, i| b.iter(|| fibonacci_fast(*i)));
//     }
//     group.finish();
// }

//criterion_group!(benches, bench_fibs);

criterion_group!(benches, cpy_benchmark);
criterion_main!(benches);
