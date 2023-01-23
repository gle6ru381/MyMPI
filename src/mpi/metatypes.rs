use crate::{shared::*, types::*, MPI_CHECK};

pub(crate) fn check_type(dtype: MPI_Datatype, comm: MPI_Comm) -> MpiResult {
    MPI_CHECK!(
        matches!(dtype, MPI_BYTE | MPI_INT | MPI_DOUBLE),
        comm,
        MPI_ERR_TYPE
    )
}

pub(crate) fn type_size(dtype: MPI_Datatype) -> Result<i32, MpiError> {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_TYPE);
    check_type(dtype, MPI_COMM_WORLD)?;

    Ok(dtype)
}
