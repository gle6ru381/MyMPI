use crate::{types::*, private::*, MPI_CHECK, MPI_CHECK_TYPE};

pub (crate) fn p_mpi_check_type(dtype : MPI_Datatype, comm : MPI_Comm) -> i32 {
    MPI_CHECK!(dtype == MPI_BYTE || dtype == MPI_INT || dtype == MPI_DOUBLE, comm, MPI_ERR_TYPE)
}

pub (crate) fn p_mpi_type_size(dtype : MPI_Datatype) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_TYPE);
    debug_assert!(dtype == MPI_BYTE || dtype == MPI_INT || dtype == MPI_DOUBLE);

    dtype
}

#[no_mangle]
pub extern "C" fn MPI_Type_size(dtype : MPI_Datatype, psize: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK_TYPE!(dtype, MPI_COMM_WORLD);
    MPI_CHECK!(!psize.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    unsafe {
        psize.write(p_mpi_type_size(dtype));
    }

    MPI_SUCCESS
}

#[no_mangle]
pub extern "C" fn MPI_Get_count(pstat : *mut MPI_Status, dtype : MPI_Datatype, pcnt : *mut i32) -> i32 {
    MPI_SUCCESS
}