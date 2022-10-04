#include <inttypes.h>
#include <stddef.h>
#define MPI_EXPORT __attribute__((visibility("default")))

typedef int32_t MPI_Datatype;
typedef int32_t MPI_Comm;
typedef int32_t MPI_Op;
typedef int32_t MPI_Errhandler;
typedef int32_t i32;

#define MPI_UNDEFINED -1
#define MPI_COMM_NULL MPI_UNDEFINED
#define MPI_COMM_SELF 0
#define MPI_COMM_WORLD 1

#define MPI_BYTE 1
#define MPI_INT 4
#define MPI_DOUBLE 8

#define MPI_MAX 0
#define MPI_MIN 1
#define MPI_SUM 2

#define MPI_ERRORS_ARE_FATAL 0
#define MPI_ERRORS_RETURN 1

#define MPI_MAX_ERROR_STRING 32

#define MPI_SUCCESS 0
#define MPI_ERR_BUFFER 1
#define MPI_ERR_COUNT 2
#define MPI_ERR_TYPE 3
#define MPI_ERR_TAG 4
#define MPI_ERR_COMM 5
#define MPI_ERR_RANK 6
#define MPI_ERR_REQUEST 7
#define MPI_ERR_ROOT 8
#define MPI_ERR_OP 9
#define MPI_ERR_ARG 10
#define MPI_ERR_UNKNOWN 11
#define MPI_ERR_TRUNCATE 12
#define MPI_ERR_OTHER 13
#define MPI_ERR_INTERN 14
#define MPI_ERR_PENDING 15
#define MPI_ERR_IN_STATUS 16
#define MPI_ERR_LASTCODE 16

extern "C" {
typedef struct _MPI_Status {
    int32_t MPI_SOURCE;
    int32_t MPI_TAG;
    int32_t MPI_ERROR;
    int32_t cnt;
} MPI_Status;

typedef struct _MPI_Request {
    void* buf;
    MPI_Status stat;
    MPI_Comm comm;
    int32_t flag;
    int32_t tag;
    int32_t cnt;
    int32_t rank;
} * MPI_Request;

MPI_EXPORT i32 MPI_Init(i32*, char***);
MPI_EXPORT i32 MPI_Finalize();
MPI_EXPORT i32 MPI_Abort(MPI_Comm, i32);
MPI_EXPORT double MPI_Wtime();
MPI_EXPORT i32 MPI_Send(const void*, i32, MPI_Datatype, i32, i32, MPI_Comm);
MPI_EXPORT i32
MPI_Recv(void*, i32, MPI_Datatype, i32, i32, MPI_Comm, MPI_Status*);
MPI_EXPORT i32 MPI_Sendrecv(
        const void*,
        i32,
        MPI_Datatype,
        i32,
        i32,
        void*,
        i32,
        MPI_Datatype,
        i32,
        i32,
        MPI_Comm,
        MPI_Status*);
MPI_EXPORT i32
MPI_Isend(const void*, i32, MPI_Datatype, i32, i32, MPI_Comm, MPI_Request*);
MPI_EXPORT i32
MPI_Irecv(void*, i32, MPI_Datatype, i32, i32, MPI_Comm, MPI_Request*);
MPI_EXPORT i32 MPI_Test(MPI_Request*, i32*, MPI_Status*);
MPI_EXPORT i32 MPI_Wait(MPI_Request*, MPI_Status*);
MPI_EXPORT i32 MPI_Waitall(i32, MPI_Request*, MPI_Status*);
MPI_EXPORT i32 MPI_Type_size(MPI_Datatype, i32*);
MPI_EXPORT i32 MPI_Get_count(MPI_Status*, MPI_Datatype, i32*);
MPI_EXPORT i32 MPI_Barrier(MPI_Comm);
MPI_EXPORT i32 MPI_Bcast(void*, i32, MPI_Datatype, i32, MPI_Comm);
MPI_EXPORT i32
MPI_Reduce(const void*, void*, i32, MPI_Datatype, MPI_Op, i32, MPI_Comm);
MPI_EXPORT i32
MPI_Allreduce(const void*, void*, i32, MPI_Datatype, MPI_Op, MPI_Comm);
MPI_EXPORT i32 MPI_Gather(
        const void*,
        i32,
        MPI_Datatype,
        void*,
        i32,
        MPI_Datatype,
        i32,
        MPI_Comm);
MPI_EXPORT i32 MPI_Allgather(
        const void*, i32, MPI_Datatype, void*, i32, MPI_Datatype, MPI_Comm);
MPI_EXPORT i32 MPI_Comm_size(MPI_Comm, i32*);
MPI_EXPORT i32 MPI_Comm_rank(MPI_Comm, i32*);
MPI_EXPORT i32 MPI_Comm_dup(MPI_Comm, MPI_Comm*);
MPI_EXPORT i32 MPI_Comm_split(MPI_Comm, i32, i32, MPI_Comm*);
MPI_EXPORT i32 MPI_Comm_get_errhandler(MPI_Comm, MPI_Errhandler*);
MPI_EXPORT i32 MPI_Comm_set_errhandler(MPI_Comm, MPI_Errhandler);
MPI_EXPORT i32 MPI_ntcpy(void* dest, const void* src, size_t size);
}
