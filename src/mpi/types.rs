pub type MPI_Datatype = i32;
pub type MPI_Comm = i32;

pub struct MPI_Status {
    pub MPI_SOURCE : i32,
    pub MPI_TAG : i32,
    pub MPI_ERROR : i32
}

pub const MPI_COMM_WORLD : i32 = 1;
pub const MPI_BYTE : i32 = 1;
pub const MPI_SUCCESS : i32 = 0;