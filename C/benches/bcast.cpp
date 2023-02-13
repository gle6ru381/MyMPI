#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <stdlib.h>
#include <string>
#include <vector>

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

    auto bcast = std::bind(
            MPI_Bcast,
            std::placeholders::_1,
            std::placeholders::_2,
            MPI_BYTE,
            0,
            MPI_COMM_WORLD);
    collective_bench(fileName, bcast);
}