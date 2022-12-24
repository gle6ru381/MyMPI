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
            eprint!("{}", format!("{}{} [{}/{}] {}\n", file_pos!(), $dbglvl, Context::rank(), Context::size() - 1, format!($fmt)));
        }
    };

    ($dbglvl:expr, $($args:tt)*) => {
        if cfg!(not(feature = "quiet")) {
            eprint!("{}", format!("{}{} [{}/{}] {}\n", file_pos!(), $dbglvl, Context::rank(), Context::size() - 1, format!($($args)*)));
        }
    }
}

#[macro_export]
macro_rules! debug_core {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgcore") {
            debug_print!(concat!("MPI Core ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgcore") {
            debug_print!(concat!("MPI Core ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_xfer {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgxfer") {
            debug_print!(concat!("MPI Xfer ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgxfer") {
            debug_print!(concat!("MPI Xfer ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_objs {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgobjects") {
            debug_print!(concat!("MPI Objects ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgobjects") {
            debug_print!(concat!("MPI Objects ", $name), $($args)*);
        }
    }
}

#[macro_export]
macro_rules! debug_bkd {
    ($name:literal, $fmt:literal) => {
        if cfg!(feature = "dbgbackend") {
            debug_print!(concat!("MPI Backend ", $name), $fmt);
        }
    };
    ($name:literal, $($args:tt)*) => {
        if cfg!(feature = "dbgbackend") {
            debug_print!(concat!("MPI Backend ", $name), $($args)*);
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
        func("Enter");
        DbgEntryExit { func }
    }
}

#[cfg(debug_assertions)]
impl<T: Fn(&'static str)> Drop for DbgEntryExit<T> {
    fn drop(&mut self) {
        (self.func)("Exit");
    }
}

#[cfg(not(debug_assertions))]
pub struct DbgEntryExit<T: Fn(&'static str)>;

#[cfg(not(debug_assertions))]
impl<T: Fn(&'static str)> DbgEntryExit<T> {
    const fn new(func: T) -> Self {
        DbgEntryExit
    }
}
