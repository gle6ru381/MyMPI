#include <mpi.h>

#define CSV_SEP ","

#include <bench_template.h>

int main(int argc, char** argv)
{
    char const* fileName;
    if (argc == 2) {
        fileName = argv[1];
    } else {
        fileName = "memcpy.csv";
    }

    auto gather = [](auto buff, auto cnt, auto mpiRank, auto mpiSize) {
        MPI_Gather(
                (char*)buff + cnt / mpiSize * mpiRank,
                cnt / mpiSize,
                MPI_BYTE,
                buff,
                cnt / mpiSize,
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);
    };
    block_bench(fileName, gather);
}