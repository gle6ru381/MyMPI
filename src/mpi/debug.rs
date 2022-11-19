#[macro_export]
macro_rules! file_pos {
    () => {
        concat!(file!(), " <", line!(), ":", column!(), ">")
    };
}

#[cfg(all(debug_assertions, not(feature = "quiet")))]
#[macro_export]
macro_rules! debug_1 {
        ($fmt:literal) => {
            if cfg!(feature = "dbglvl1") {
                {
                    let s = format!(concat!(file_pos!(), " [{}/{}] "), Context::rank(), Context::size() - 1);
                    let f = format!($fmt);
                    eprintln!("{s}{f}");
                }
            }
        };
        ($($args:tt)*) => {
            if cfg!(feature = "dbglvl1") {
                {
                    let s = format!(concat!(file_pos!(), " [{}/{}] "), Context::rank(), Context::size() - 1);
                    let f = format!($($args)*);
                    eprintln!("{s}{f}");
                }
            }
        };
}

#[cfg(all(debug_assertions, not(feature = "quiet")))]
#[macro_export]
macro_rules! debug {
    ($fmt:literal) => {
        if cfg!(feature = "dbglvl3") {
            {
                let s = format!(concat!(file_pos!(), " [{}/{}] "), Context::rank(), Context::size() - 1);
                let f = format!($fmt);
                eprintln!("{s}{f}");
            }
        }
    };
    ($($args:tt)*) => {
        if cfg!(feature = "dbglvl3") {
            {
                let s = format!(concat!(file_pos!(), " [{}/{}] "), Context::rank(), Context::size() - 1);
                let f = format!($($args)*);
                eprintln!("{s}{f}");
            }
        }
    };
}

#[cfg(any(not(debug_assertions), feature = "quiet"))]
#[macro_export]
macro_rules! debug {
    ($fmt:literal) => {};
    ($($args:tt)*) => {};
}

#[cfg(any(not(debug_assertions), feature = "quiet"))]
#[macro_export]
macro_rules! debug_1 {
    ($fmt:literal) => {};
    ($($args:tt)*) => {};
}
