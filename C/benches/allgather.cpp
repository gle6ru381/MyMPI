#include <mpi.h>

#define CSV_SEP ","

#include <bench_template.h>

int main(int argc, char** argv)
{
    char const* fileName;
    if (argc >= 2) {
        fileName = argv[1];
    } else {
        fileName = "memcpy.csv";
    }

    if (argc == 3 && argv[2] != "0") {
        setenv("MPI_SIZE", argv[2], 1);
    }

    auto allgather = [](auto buff, auto cnt, auto mpiRank, auto mpiSize) {
        MPI_Allgather(
                (char*)buff + cnt / mpiSize * mpiRank,
                cnt / mpiSize,
                MPI_BYTE,
                buff,
                cnt / mpiSize,
                MPI_BYTE,
                MPI_COMM_WORLD);
    };
    block_bench(fileName, allgather);
}