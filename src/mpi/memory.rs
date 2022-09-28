use std::arch::asm;
use std::ffi::c_void;

pub fn ntcpy(mut dest : *mut c_void, mut src : *const c_void, mut size : usize) {
    unsafe {
        while size >= 256 {
            asm!(
                "vmovntdqa {temp0}, [{src} + 0]",
                "vmovntdqa {temp1}, [{src} + 32]",
                "vmovntdqa {temp2}, [{src} + 64]",
                "vmovntdqa {temp3}, [{src} + 96]",
                "vmovntdqa {temp4}, [{src} + 128]",
                "vmovntdqa {temp5}, [{src} + 160]",
                "vmovntdqa {temp6}, [{src} + 192]",
                "vmovntdqa {temp7}, [{src} + 224]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 32], {temp1}",
                "vmovntdq [{dest} + 64], {temp2}",
                "vmovntdq [{dest} + 96], {temp3}",
                "vmovntdq [{dest} + 128], {temp4}",
                "vmovntdq [{dest} + 160], {temp5}",
                "vmovntdq [{dest} + 192], {temp6}",
                "vmovntdq [{dest} + 224], {temp7}",
            //    "add {dest}, 256",
            //    "add {src}, 256",
            //    "sub {size:e}, 256",
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
            //    size = inout(reg) size,
            );
            dest = dest.add(256);
            src = src.add(256);
            size -= 256;
        }
        if size >= 128 {
            asm!(
                "vmovntdqa {temp0}, [{src} + 0]",
                "vmovntdqa {temp1}, [{src} + 32]",
                "vmovntdqa {temp2}, [{src} + 64]",
                "vmovntdqa {temp3}, [{src} + 96]",
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
            dest = dest.add(256);
            src = src.add(256);
            size -= 128;
        }
        if size >= 64 {
            asm!(
                "vmovntdqa {temp0}, [{src} + 0]",
                "vmovntdqa {temp1}, [{src} + 32]",
                "vmovntdq [{dest} + 0], {temp0}",
                "vmovntdq [{dest} + 32], {temp1}",
                dest = in(reg) dest,
                src = in(reg) src,
                temp0 = out(ymm_reg) _,
                temp1 = out(ymm_reg) _,
            );
            dest = dest.add(64);
            src  = src.add(64);
            size -= 64;
        }
        if size >= 32 {
            asm!(
                "vmovntdqa {temp0}, [{src} + 0]",
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
                "vmovntdqa {temp0}, [{src} + 0]",
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
            size -= 1;
        }
    }
}

pub fn xmmntcpy(mut dest : *mut c_void, mut src : *const c_void, mut n : usize) {
    unsafe {
        while n >= 128 {
            asm!(
                "movntdqa {temp0}, [{src} + 0]",
                "movntdqa {temp1}, [{src} + 16]",
                "movntdqa {temp2}, [{src} + 32]",
                "movntdqa {temp3}, [{src} + 48]",
                "movntdqa {temp4}, [{src} + 64]",
                "movntdqa {temp5}, [{src} + 80]",
                "movntdqa {temp6}, [{src} + 96]",
                "movntdqa {temp7}, [{src} + 112]",
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
            n -= 128;
        }
        if n >= 64 {
            asm!(
                "movntdqa {temp0}, [{src} + 0]",
                "movntdqa {temp1}, [{src} + 16]",
                "movntdqa {temp2}, [{src} + 32]",
                "movntdqa {temp3}, [{src} + 48]",
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
                "movntdqa {temp0}, [{src} + 0]",
                "movntdqa {temp1}, [{src} + 16]",
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
                "movntdqa {temp0}, [{src} + 0]",
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
            n -= 1;
        }
    }
}