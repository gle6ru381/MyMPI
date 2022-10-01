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

fn procWithArray(vec: &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[inline(never)]
fn fillArray<'a>(vec: &'a mut Vec<i32>) {
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

fn bytes_format(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}b")
    } else if bytes < 1024 * 1024 {
        format!("{}Kb", bytes / 1024)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{}Mb", bytes / (1024 * 1024))
    } else {
        format!("{}Gb", bytes / (1024 * 1024 * 1024))
    }
}

fn cpy_benchmark(c: &mut Criterion) {
    c.without_output();
    let buff_vec = vec![
        4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864,
    ];
    let vec_buff = vec![
        4096, 16384, 131072, 262144, 524288, 786432, 1048576, 2097152, 4194304, 6291456, 7876608,
        8388608,
    ];
    for &vec_size in vec_buff.iter() {
        let mut g = c.benchmark_group(format!("Vector size: {}", bytes_format(vec_size)));
        g.noise_threshold(0.05);
        g.significance_level(0.0001);
        g.confidence_level(0.99999);
        g.warm_up_time(Duration::from_nanos(1));
        g.measurement_time(Duration::from_secs(20));
        g.plot_config(
            PlotConfiguration::default()
                .y_scale(criterion::AxisScale::Logarithmic)
                .x_scale(criterion::AxisScale::Logarithmic)
                .tics(buff_vec.clone())
                .x_label(format!("Buffer size"))
                .x_grid_major(true)
                .y_grid_major(true),
        );
        g.sample_size(5);
        g.sampling_mode(criterion::SamplingMode::Linear);
        let mut vec = createArray(vec_size);
        for &size in buff_vec.iter() {
            //let mut data = allocate_xmm(size);

            // g.throughput(criterion::Throughput::Bytes((vec_size + size) as u64));
            // g.bench_with_input(BenchmarkId::new("128bit", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         xmmntcpy(data.1, data.0, *size);
            //         procWithArray(&mut vec);
            //     })
            // });
            // dealoc_xmm(size, data.0, data.1);

            let vec_ptr: *mut Vec<i32> = &mut vec;

            let mut data = allocate_default(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Default", size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );
            dealoc_default(size as usize, data.0, data.1);

            data = allocate_ymm(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Non temporal", size),
                &size,
                |x, size| {
                    x.iter(|| {
                        ymmntcpy(data.1, data.0, *size as usize);
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );

            g.bench_with_input_prepare(
                BenchmarkId::new("Default align", size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );

            // g.bench_with_input_prepare(
            //     BenchmarkId::new("256bit_prefetch", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             ymmntcpy_prefetch(data.1, data.0, *size as usize);
            //             procWithArray(&mut vec);
            //         })
            //     },
            //     |_, _| {
            //         fillArray(unsafe { &mut *vec_ptr });
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

            // g.bench_with_input_prepare(
            //     BenchmarkId::new("256bit_aligned", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             ymmntcpy_aligned(data.1, data.0, *size as usize);
            //             procWithArray(&mut vec);
            //         })
            //     },
            //     |_, _| {
            //         fillArray(unsafe { &mut *vec_ptr });
            //     },
            // );

            // g.bench_with_input(BenchmarkId::new("256bit_short_prefetch_aligned", size), &size, |x, size| {
            //     x.iter(|| {
            //         fillArray(&mut vec);
            //         ymmntcpy_short_prefetch_aligned(data.1, data.0, *size as usize);
            //         procWithArray(&mut vec);
            //     })
            // });

            // g.bench_with_input_prepare(
            //     BenchmarkId::new("256bit_prefetch_aligned", size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             ymmntcpy_prefetch_aligned(data.1, data.0, *size as usize);
            //             procWithArray(&mut vec);
            //         })
            //     },
            //     |_, _| {
            //         fillArray(unsafe { &mut *vec_ptr });
            //     },
            // );
            dealoc_ymm(size as usize, data.0, data.1);

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

    for &size in buff_vec.iter() {
        let mut g = c.benchmark_group(format!("Buffer size: {}", bytes_format(size as usize)));
        g.noise_threshold(0.05);
        g.significance_level(0.0001);
        g.confidence_level(0.99999);
        g.warm_up_time(Duration::from_nanos(1));
        g.sampling_mode(criterion::SamplingMode::Linear);
        g.measurement_time(Duration::from_secs(20));
        g.plot_config(
            PlotConfiguration::default()
                .y_scale(criterion::AxisScale::Logarithmic)
                .x_scale(criterion::AxisScale::Logarithmic)
                .tics(buff_vec.clone())
                .x_label(format!("Vector size"))
                .x_grid_major(true)
                .y_grid_major(true),
        );
        g.sample_size(5);
        for &vec_size in vec_buff.iter() {
            //let mut data = allocate_xmm(size);
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
            let vec_ptr: *mut Vec<i32> = &mut vec;

            let mut data = allocate_ymm(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Non temporal", vec_size),
                &size,
                |x, size| {
                    x.iter(|| {
                        ymmntcpy(data.1, data.0, *size as usize);
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );
            dealoc_ymm(size as usize, data.0, data.1);

            data = allocate_default(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Default", vec_size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );
            dealoc_default(size as usize, data.0, data.1);

            data = allocate_ymm(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Default align", vec_size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        procWithArray(&mut vec);
                    })
                },
                |_, _| {
                    fillArray(unsafe { &mut *vec_ptr });
                },
            );

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

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_short_prefetch_aligned", vec_size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_short_prefetch_aligned(data.1, data.0, *size as usize);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );

            // g.bench_with_input(
            //     BenchmarkId::new("256bit_prefetch_aligned", vec_size),
            //     &size,
            //     |x, size| {
            //         x.iter(|| {
            //             fillArray(&mut vec);
            //             ymmntcpy_prefetch_aligned(data.1, data.0, *size as usize);
            //             procWithArray(&mut vec);
            //         })
            //     },
            // );
            dealoc_ymm(size as usize, data.0, data.1);

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
