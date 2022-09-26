#[macro_export]
macro_rules! file_pos {
    () => {
        concat!(file!(), " <", line!(), ":", column!(), ">")
    };
}

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