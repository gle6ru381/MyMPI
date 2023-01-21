#[allow(unused_imports)]
use std::marker::PhantomData;

#[macro_export]
macro_rules! file_pos {
    () => {
        if cfg!(feature = "dbgfilepos") {
            concat!(file!(), " <", line!(), ":", column!(), ">: ")
        } else {
            ""
        }
    };
}

#[macro_export]
macro_rules! debug_print {
    ($dbglvl:expr, $fmt:literal) => {
        if cfg!(not(feature = "quiet")) {
            eprint!("{}", format!("{}{} [{}/{}] {}\n", crate::file_pos!(), $dbglvl, crate::context::Context::rank(), crate::context::Context::size() - 1, format!($fmt)));
        }
    };

    ($dbglvl:expr, $($args:tt)*) => {
        if cfg!(not(feature = "quiet")) {
            eprint!("{}", format!("{}{} [{}/{}] {}\n", crate::file_pos!(), $dbglvl, crate::context::Context::rank(), crate::context::Context::size() - 1, format!($($args)*)));
        }
    }
}

#[macro_export]
macro_rules! debug_core {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgcore") {
            crate::debug_print!(concat!("MPI Core ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgcore") {
            crate::debug_print!(concat!("MPI Core ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_xfer {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgxfer") {
            crate::debug_print!(concat!("MPI Xfer ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgxfer") {
            crate::debug_print!(concat!("MPI Xfer ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_objs {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgobjects") {
            crate::debug_print!(concat!("MPI Objects ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgobjects") {
            crate::debug_print!(concat!("MPI Objects ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_bkd {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgbackend") {
            crate::debug_print!(concat!("MPI Backend ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgbackend") {
            crate::debug_print!(concat!("MPI Backend ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_coll {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgcoll") {
            crate::debug_print!(concat!("MPI Collective ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgcoll") {
            crate::debug_print!(concat!("MPI Collective ", $name), $($args)*);
        }
    }
}

#[cfg(debug_assertions)]
pub struct DbgEntryExit<T: Fn(&'static str)> {
    func: T,
}

#[cfg(debug_assertions)]
impl<T: Fn(&'static str)> DbgEntryExit<T> {
    pub fn new(func: T) -> Self {
        if cfg!(dbgentryfn) {
            func("Enter");
        }
        DbgEntryExit { func }
    }
}

#[cfg(debug_assertions)]
impl<T: Fn(&'static str)> Drop for DbgEntryExit<T> {
    fn drop(&mut self) {
        if cfg!(dbgentryfn) {
            (self.func)("Exit");
        }
    }
}

#[cfg(not(debug_assertions))]
pub struct DbgEntryExit<T: Fn(&'static str)> {
    phantom: PhantomData<T>,
}

#[cfg(not(debug_assertions))]
impl<T: Fn(&'static str)> DbgEntryExit<T> {
    pub fn new(_: T) -> Self {
        DbgEntryExit {
            phantom: PhantomData,
        }
    }
}
