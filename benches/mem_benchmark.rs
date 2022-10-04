use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use mpi::{memory::*, *};
use std::alloc::*;
use std::slice::from_raw_parts_mut;
use std::time::Duration;

fn create_array(size: usize) -> Vec<i32> {
    let mut vec = Vec::with_capacity(size / 4);
    for i in 0..vec.capacity() {
        vec.push(i as i32);
    }
    vec
}

fn proc_with_array(vec: &mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[inline(never)]
fn fill_array<'a>(vec: &'a mut Vec<i32>) {
    for i in 0..vec.len() {
        vec[i] = i as i32;
    }
}

#[allow(dead_code)]
fn allocate_ymm(size: usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        let b = alloc(layout) as *mut i32;
        for (i, val) in a.iter_mut().enumerate() {
            *val = (i * 4) as i32;
        }
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

#[allow(dead_code)]
fn dealoc_ymm(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn dealoc_xmm(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 16).unwrap();
        dealloc(a as *mut u8, layout);
        dealloc(b as *mut u8, layout);
    }
}

#[allow(dead_code)]
fn allocate_default(size: usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size + 1, 1).unwrap();
        let a = from_raw_parts_mut(alloc(layout).add(1) as *mut i32, size / 4);
        let b = alloc(layout).add(1) as *mut i32;
        for (i, val) in a.iter_mut().enumerate() {
            *val = (i * 4) as i32;
        }
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

#[allow(dead_code)]
fn dealoc_default(size: usize, a: *mut c_void, b: *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size + 1, 1).unwrap();
        dealloc((a as *mut u8).sub(1), layout);
        dealloc((b as *mut u8).sub(1), layout);
    }
}

#[allow(dead_code)]
fn allocate_avx512(size: usize) -> (*mut c_void, *mut c_void) {
    unsafe {
        let layout = Layout::from_size_align(size, 32).unwrap();
        let a = from_raw_parts_mut(alloc(layout) as *mut i32, size / 4);
        let b = alloc(layout) as *mut i32;
        (a.as_mut_ptr() as *mut c_void, b as *mut c_void)
    }
}

#[allow(dead_code)]
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
        g.sample_size(250);
        g.sampling_mode(criterion::SamplingMode::Linear);
        let mut vec = create_array(vec_size);
        for &size in buff_vec.iter() {
            let vec_ptr: *mut Vec<i32> = &mut vec;

            let mut data = allocate_default(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Default", size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
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
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
                },
            );

            g.bench_with_input_prepare(
                BenchmarkId::new("Default align", size),
                &size,
                |x, size| {
                    x.iter(|| {
                        unsafe { std::ptr::copy(data.1, data.0, *size as usize) };
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
                },
            );

            dealoc_ymm(size as usize, data.0, data.1);
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
        g.sample_size(250);
        for &vec_size in vec_buff.iter() {
            let mut vec = create_array(vec_size);

            let vec_ptr: *mut Vec<i32> = &mut vec;

            let mut data = allocate_ymm(size as usize);
            g.bench_with_input_prepare(
                BenchmarkId::new("Non temporal", vec_size),
                &size,
                |x, size| {
                    x.iter(|| {
                        ymmntcpy(data.1, data.0, *size as usize);
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
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
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
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
                        proc_with_array(&mut vec);
                    })
                },
                |_, _| {
                    fill_array(unsafe { &mut *vec_ptr });
                },
            );
            dealoc_ymm(size as usize, data.0, data.1);
        }
        g.finish();
    }
}

criterion_group!(benches, cpy_benchmark);
criterion_main!(benches);
