use std::{
    fmt::Display,
    ops::AddAssign,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use crate::shared::*;

fn sum_proc<T>(src: *const T, dst: *mut T, len: usize)
where
    T: AddAssign + Copy,
{
    unsafe {
        let d = from_raw_parts_mut(dst, len);
        let s = from_raw_parts(src, len);
        for i in 0..len {
            d[i] += s[i];
        }
    }
}

fn min_proc<T>(src: *const T, dst: *mut T, len: usize)
where
    T: PartialOrd + Copy,
{
    unsafe {
        let d = from_raw_parts_mut(dst, len);
        let s = from_raw_parts(src, len);
        for i in 0..len {
            if s[i] < d[i] {
                d[i] = s[i];
            }
        }
    }
}

fn max_proc<T>(src: *const T, dst: *mut T, len: usize)
where
    T: PartialOrd + Copy + Display,
{
    unsafe {
        let d = from_raw_parts_mut(dst, len);
        let s = from_raw_parts(src, len);
        for i in 0..len {
            if s[i] > d[i] {
                d[i] = s[i];
            }
        }
    }
}

pub fn sum(src: *const c_void, dst: *mut c_void, len: i32, dtype: MPI_Datatype) {
    match dtype {
        MPI_BYTE => sum_proc(src as *const i8, dst as *mut i8, len as usize),
        MPI_INT => sum_proc(src as *const i32, dst as *mut i32, len as usize),
        MPI_DOUBLE => sum_proc(src as *const f64, dst as *mut f64, len as usize),
        _ => unreachable!(),
    }
}

pub fn min(src: *const c_void, dst: *mut c_void, len: i32, dtype: MPI_Datatype) {
    match dtype {
        MPI_BYTE => min_proc(src as *const i8, dst as *mut i8, len as usize),
        MPI_INT => min_proc(src as *const i32, dst as *mut i32, len as usize),
        MPI_DOUBLE => min_proc(src as *const f64, dst as *mut f64, len as usize),
        _ => unreachable!(),
    }
}

pub fn max(src: *const c_void, dst: *mut c_void, len: i32, dtype: MPI_Datatype) {
    match dtype {
        MPI_BYTE => max_proc(src as *const i8, dst as *mut i8, len as usize),
        MPI_INT => max_proc(src as *const i32, dst as *mut i32, len as usize),
        MPI_DOUBLE => max_proc(src as *const f64, dst as *mut f64, len as usize),
        _ => unreachable!(),
    }
}
