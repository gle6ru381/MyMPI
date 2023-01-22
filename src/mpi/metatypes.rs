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

#[no_mangle]
pub extern "C" fn MPI_Type_size(dtype: MPI_Datatype, psize: *mut i32) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(!psize.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    unsafe {
        return match type_size(dtype) {
            Ok(size) => {
                psize.write(size);
                MPI_SUCCESS
            }
            Err(code) => code as i32,
        };
    }
}

#[no_mangle]
pub extern "C" fn MPI_Get_count(
    pstat: *const MPI_Status,
    dtype: MPI_Datatype,
    pcnt: *mut i32,
) -> i32 {
    MPI_CHECK!(Context::is_init(), MPI_COMM_WORLD, MPI_ERR_OTHER);
    MPI_CHECK!(!pstat.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);
    MPI_CHECK!(!pcnt.is_null(), MPI_COMM_WORLD, MPI_ERR_ARG);

    return match type_size(dtype) {
        Ok(size) => unsafe {
            pcnt.write((*pstat).cnt / size);
            MPI_SUCCESS
        },
        Err(code) => code as i32,
    };
}
