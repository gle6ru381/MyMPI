use crate::types::*;

#[no_mangle]
pub extern "C" fn MPI_Type_size(dtype : MPI_Datatype, psize: *mut i32) -> i32 {
    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Get_count(pstat : *mut MPI_Status, dtype : MPI_Datatype, pcnt : *mut i32) -> i32 {
    MPI_SUCCESS
}