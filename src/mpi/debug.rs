#[macro_export]
macro_rules! file_pos {
    () => {
        concat!(file!(), " <", line!(), ":", column!(), ">")
    };
}

#[cfg(all(debug_assertions, not(feature = "quiet")))]
#[macro_export]
macro_rules! debug {
    ($fmt:literal) => {
        print!(concat!(file_pos!(), " "));
        print!("[{}/{}] ", Context::rank(), Context::size() - 1);
        println!($fmt)
    };
    ($($args:tt)*) => {
        print!(concat!(file_pos!(), " "));
        print!("[{}/{}] ", Context::rank(), Context::size() - 1);
        println!($($args)*);
    };
}

#[cfg(any(not(debug_assertions), feature = "quiet"))]
#[macro_export]
macro_rules! debug {
    ($fmt:literal) => {};
    ($($args:tt)*) => {};
}
