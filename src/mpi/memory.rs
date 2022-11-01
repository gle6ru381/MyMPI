#![feature(asm)]
use std::arch::asm;
use std::ffi::c_void;

#[cfg(target_feature = "sse2")]
pub fn sse2_ntcpy(mut dest: *mut c_void, mut src: *const c_void, mut size: usize) {
    unsafe {
        if dest as usize % 32 != 0 || src as usize % 32 != 0 {
            if dest as usize % 32 == src as usize % 32 {
                while dest as usize % 32 != 0 {
                    *(dest as *mut u8) = *(src as *const u8);
                    dest = dest.add(1);
                    src = src.add(1);
                    size -= 1;
                }
            } else {
                std::ptr::copy(src, dest, size);
                return;
            }
        }
        while size >= 128 {
            asm!(
                "prefetchnta [{src} + 0]",
                "prefetchnta [{src} + 64]",
                "movdqa {temp0}, [{src} + 0]",
                "movdqa {temp1}, [{src} + 16]",
                "movdqa {temp2}, [{src} + 32]",
                "movdqa {temp3}, [{src} + 48]",
                "movdqa {temp4}, [{src} + 64]",
                "movdqa {temp5}, [{src} + 80]",
                "movdqa {temp6}, [{src} + 96]",
                "movdqa {temp7}, [{src} + 112]",
                "movntdq [{dest} + 0], {temp0}",
                "movntdq [{dest} + 16], {temp1}",
                "movntdq [{dest} + 32], {temp2}",
                "movntdq [{dest} + 48], {temp3}",
                "movntdq [{dest} + 64], {temp4}",
                "movntdq [{dest} + 80], {temp5}",
                "movntdq [{dest} + 96], {temp6}",
                "movntdq [{dest} + 112], {temp7}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
                temp2 = out(xmm_reg) _,
                temp3 = out(xmm_reg) _,
                temp4 = out(xmm_reg) _,
                temp5 = out(xmm_reg) _,
                temp6 = out(xmm_reg) _,
                temp7 = out(xmm_reg) _,
            );
            dest = dest.add(128);
            src = src.add(128);
            size -= 128;
        }
        if size >= 64 {
            asm!(
                "prefetchnta [{src} + 0]",
                "movdqa {temp0}, [{src} + 0]",
                "movdqa {temp1}, [{src} + 16]",
                "movdqa {temp2}, [{src} + 32]",
                "movdqa {temp3}, [{src} + 48]",
                "movntdq [{dest} + 0], {temp0}",
                "movntdq [{dest} + 16], {temp1}",
                "movntdq [{dest} + 32], {temp2}",
                "movntdq [{dest} + 48], {temp3}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
                temp2 = out(xmm_reg) _,
                temp3 = out(xmm_reg) _,
            );
            dest = dest.add(64);
            src = src.add(64);
            size -= 64;
        }
        if size >= 32 {
            asm!(
                "prefetchnta [{src} + 0]",
                "movdqa {temp0}, [{src} + 0]",
                "movdqa {temp1}, [{src} + 16]",
                "movntdq [{dest} + 0], {temp0}",
                "movntdq [{dest} + 16], {temp1}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
            );
            dest = dest.add(32);
            src = src.add(32);
            size -= 32;
        }
        if size >= 16 {
            asm!(
                "movdqa {temp0}, [{src} + 0]",
                "movntdq [{dest} + 0], {temp0}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
            );
            dest = dest.add(16);
            src = src.add(16);
            size -= 16;
        }
        while size != 0 {
            *(dest as *mut u8) = *(src as *const u8);
            dest = dest.add(1);
            src = src.add(1);
            size -= 1;
        }
    }
}

#[cfg(target_feature = "avx2")]
pub fn avx2_ntcpy(mut dest: *mut c_void, mut src: *const c_void, mut size: usize) {
    unsafe {
        if dest as usize % 32 != 0 || src as usize % 32 != 0 {
            if dest as usize % 32 == src as usize % 32 {
                while dest as usize % 32 != 0 {
                    *(dest as *mut u8) = *(src as *const u8);
                    dest = dest.add(1);
                    src = src.add(1);
                    size -= 1;
                }
            } else {
                std::ptr::copy(src, dest, size);
                return;
            }
        }
        debug_assert!(dest as usize % 32 == 0);
        debug_assert!(src as usize % 32 == 0);
        while size >= 256 {
            asm!(
                "prefetchnta [{src} + 0]",
                "prefetchnta [{src} + 64]",
                "prefetchnta [{src} + 128]",
                "prefetchnta [{src} + 192]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 32]",
                "vmovdqa {temp2}, [{src} + 64]",
                "vmovdqa {temp3}, [{src} + 96]",
                "vmovdqa {temp4}, [{src} + 128]",
                "vmovdqa {temp5}, [{src} + 160]",
                "vmovdqa {temp6}, [{src} + 192]",
                "vmovdqa {temp7}, [{src} + 224]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 32], {temp1}",
                "vmovntdq [{dest} + 64], {temp2}",
                "vmovntdq [{dest} + 96], {temp3}",
                "vmovntdq [{dest} + 128], {temp4}",
                "vmovntdq [{dest} + 160], {temp5}",
                "vmovntdq [{dest} + 192], {temp6}",
                "vmovntdq [{dest} + 224], {temp7}",
                dest = inout(reg) dest,
                src = inout(reg) src,
                temp0 = out(ymm_reg) _,
                temp1 = out(ymm_reg) _,
                temp2 = out(ymm_reg) _,
                temp3 = out(ymm_reg) _,
                temp4 = out(ymm_reg) _,
                temp5 = out(ymm_reg) _,
                temp6 = out(ymm_reg) _,
                temp7 = out(ymm_reg) _,
            );
            dest = dest.add(256);
            src = src.add(256);
            size -= 256;
        }
        if size >= 128 {
            asm!(
                "prefetchnta [{src} + 0]",
                "prefetchnta [{src} + 64]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 32]",
                "vmovdqa {temp2}, [{src} + 64]",
                "vmovdqa {temp3}, [{src} + 96]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 32], {temp1}",
                "vmovntdq [{dest} + 64], {temp2}",
                "vmovntdq [{dest} + 96], {temp3}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(ymm_reg) _,
                temp1 = out(ymm_reg) _,
                temp2 = out(ymm_reg) _,
                temp3 = out(ymm_reg) _,
            );
            dest = dest.add(128);
            src = src.add(128);
            size -= 128;
        }
        if size >= 64 {
            asm!(
                "prefetchnta [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 32]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 32], {temp1}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(ymm_reg) _,
                temp1 = out(ymm_reg) _,
            );
            dest = dest.add(64);
            src = src.add(64);
            size -= 64;
        }
        if size >= 32 {
            asm!(
                "prefetchnta [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovntdq [{dest} + 0], {temp0}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(ymm_reg) _,
            );
            dest = dest.add(32);
            src = src.add(32);
            size -= 32;
        }
        if size >= 16 {
            asm!(
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovntdq [{dest} + 0], {temp0}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
            );
            dest = dest.add(16);
            src = src.add(16);
            size -= 16;
        }
        while size != 0 {
            *(dest as *mut u8) = *(src as *const u8);
            dest = dest.add(1);
            src = src.add(1);
            size -= 1;
        }
        asm!("sfence");
    }
}

#[cfg(target_feature = "avx")]
pub fn avx_ntcpy(mut dest: *mut c_void, mut src: *const c_void, mut n: usize) {
    unsafe {
        while n >= 128 {
            asm!(
                "prefetchnta [{src} + 0]",
                "prefetchnta [{src} + 64]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 16]",
                "vmovdqa {temp2}, [{src} + 32]",
                "vmovdqa {temp3}, [{src} + 48]",
                "vmovdqa {temp4}, [{src} + 64]",
                "vmovdqa {temp5}, [{src} + 80]",
                "vmovdqa {temp6}, [{src} + 96]",
                "vmovdqa {temp7}, [{src} + 112]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 16], {temp1}",
                "vmovntdq [{dest} + 32], {temp2}",
                "vmovntdq [{dest} + 48], {temp3}",
                "vmovntdq [{dest} + 64], {temp4}",
                "vmovntdq [{dest} + 80], {temp5}",
                "vmovntdq [{dest} + 96], {temp6}",
                "vmovntdq [{dest} + 112], {temp7}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
                temp2 = out(xmm_reg) _,
                temp3 = out(xmm_reg) _,
                temp4 = out(xmm_reg) _,
                temp5 = out(xmm_reg) _,
                temp6 = out(xmm_reg) _,
                temp7 = out(xmm_reg) _,
            );
            dest = dest.add(128);
            src = src.add(128);
            n -= 128;
        }
        if n >= 64 {
            asm!(
                "prefetchnta [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 16]",
                "vmovdqa {temp2}, [{src} + 32]",
                "vmovdqa {temp3}, [{src} + 48]",
                "movntdq [{dest} + 0], {temp0}",
                "movntdq [{dest} + 16], {temp1}",
                "movntdq [{dest} + 32], {temp2}",
                "movntdq [{dest} + 48], {temp3}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
                temp2 = out(xmm_reg) _,
                temp3 = out(xmm_reg) _,
            );
            dest = dest.add(64);
            src = src.add(64);
            n -= 64;
        }
        if n >= 32 {
            asm!(
                "prefetchnta [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp1}, [{src} + 16]",
                "movntdq [{dest} + 0], {temp0}",
                "movntdq [{dest} + 16], {temp1}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
                temp1 = out(xmm_reg) _,
            );
            dest = dest.add(32);
            src = src.add(32);
            n -= 32;
        }
        if n >= 16 {
            asm!(
                "vmovdqa {temp0}, [{src} + 0]",
                "movntdq [{dest} + 0], {temp0}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(xmm_reg) _,
            );
            dest = dest.add(16);
            src = src.add(16);
            n -= 16;
        }
        while n != 0 {
            *(dest as *mut u8) = *(src as *const u8);
            dest = dest.add(1);
            src = src.add(1);
            n -= 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub fn avx512ntcpy(mut dest: *mut c_void, mut src: *mut c_void, size: usize) {
    unsafe {
        while size >= 128 {
            asm!(
                "prefetchnta [{src} + 0]",
                "prefetchnta [{src} + 64]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 64]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 64], {temp1}",
                dest = inout(reg) dest,
                src = inout(reg) src,
                temp0 = out(zmm_reg) _,
                temp1 = out(zmm_reg) _,
            );
            dest = dest.add(128);
            src = src.add(128);
            size -= 128;
        }
        if size >= 64 {
            asm!(
                "prefetchnta [{src} + 0]",
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovntdq [{dest} + 0], {temp0}",
                dest = inout(reg) dest,
                src = inout(reg) src,
                temp0 = out(zmm_reg) _,
            );
            dest = dest.add(64);
            src = src.add(64);
            size -= 64;
        }
        if size >= 32 {
            asm!(
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovntdq [{dest} + 0], {temp0}",
                dest = inout(reg) dest,
                src = inout(reg) src,
                temp0 = out(ymm_reg) _,
            );
            dest = dest.add(32);
            src = src.add(32);
            size -= 32;
        }
        if size >= 16 {
            asm!(
                "vmovdqa {temp0}, [{src} + 0]",
                "vmovntdq [{dest} + 0], {temp0}",
                dest = inout(reg) dest,
                src = inout(reg) src,
                temp0 = out(xmm_reg) _,
            );
            dest = dest.add(16);
            src = src.add(16);
            size -= 32;
        }
        while size != 0 {
            *(dest as *mut u8) = *(src as *const u8);
            dest = dest.add(1);
            src = src.add(1);
            size -= 1;
        }
    }
}

pub fn memcpy(dest: *mut c_void, src: *const c_void, size: usize) {
    if cfg!(feature = "ntcpy") {
        if size == 0 {
            return;
        }
        ymmntcpy(dest, src, size);
        //MPI_ntcpy(dest, src, size);
    } else {
        unsafe { std::ptr::copy(src, dest, size) };
    }
}

#[cfg(target_feature = "avx512")]
pub extern "C" fn MPI_ntcpy(dest: *mut c_void, src: *const c_void, size: usize) {
    avx512ntcpy(dest, src, size);
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512")))]
pub extern "C" fn MPI_ntcpy(dest: *mut c_void, src: *const c_void, size: usize) {
    ymmntcpy(dest, src, size);
}

#[cfg(all(not(target_feature = "avx2"), not(target_feature = "avx512")))]
pub extern "C" fn MPI_ntcpy(dest: *mut c_void, src: *const c_void, size: usize) {
    std::ptr::copy(src, dest, size)
}
