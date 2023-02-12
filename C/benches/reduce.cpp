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

    auto tmpbuff = new char[bench_max_size() / 8];
    auto reduce = [tmpbuff](auto buff, auto cnt, auto mpiRank, auto mpiSize) {
        MPI_Reduce(
                (char*)buff + cnt / mpiSize * mpiRank,
                tmpbuff,
                cnt / mpiSize,
                MPI_BYTE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD);
    };
    block_bench(fileName, reduce);
}