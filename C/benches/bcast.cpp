#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <memory>
#include <mpi.h>
#include <string>

#define CSV_SEP " , "

#include "bench_template.h"

int main(int argc, char** argv)
{
    char const* fileName;
    if (argc == 2) {
        fileName = argv[1];
    } else {
        fileName = "memcpy.csv";
    }

    auto bcast = std::bind(MPI_Bcast, std::placeholders::_1, std::placeholders::_2, MPI_BYTE, 0, MPI_COMM_WORLD);
    collective_bench(fileName, bcast);
}