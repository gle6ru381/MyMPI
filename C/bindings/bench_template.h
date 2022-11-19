#include "mpi.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#ifndef CSV_SEP
#define CSV_SEP ","
#endif

template <typename T, size_t N>
T median(std::array<T, N> const& arr)
{
    return arr.size() % 2 == 0
            ? (arr[arr.size() / 2] + arr[arr.size() / 2 + 1]) / 2
            : arr[arr.size() / 2 + 1];
}

template <typename Predicate>
void collective_bench(char const* fName, Predicate func)
{
    using clock = std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using unit = std::chrono::nanoseconds;

    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string fileName = fName;
    auto slash = fileName.rfind('/');
    auto dot = fileName.rfind('.');
    if (dot == std::string::npos || dot < slash) {
        fileName += std::to_string(rank);
    } else {
        fileName.insert(dot, std::to_string(rank));
    }

    constexpr int vec_sizes[]
            = {125952,
               204800,
               256000,
               458752,
               921600,
               1572864,
               3932160,
               6291456,
               13631488,
               24641536};
    constexpr int msg_sizes[]
            = {125952,
               204800,
               256000,
               458752,
               921600,
               1572864,
               3932160,
               6291456,
               13631488,
               24641536};

    constexpr int nsamples = 50;
    std::array<long, nsamples> fAccess;
    std::array<long, nsamples> sAccess;
    std::array<long, nsamples> sendTimes;

    auto csvFile = fopen(fileName.c_str(), "w");
    fprintf(csvFile,
            "Vector size" CSV_SEP "Message size" CSV_SEP "unit" CSV_SEP
            "First access "
            "min" CSV_SEP "Second access min" CSV_SEP
            "Access slowdown "
            "min" CSV_SEP "Access diff" CSV_SEP
            "Send "
            "time min\n");

    for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
        auto const vec_size = vec_sizes[vec_idx] / 8;
        auto vec = std::vector<long>(vec_size);

        for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes); msg_idx++) {
            for (int sample = 0; sample < nsamples; sample++) {
                long tmpVal;
                auto data = vec.data();
                for (auto l = 0; l < vec_size; l++) {
                    asm volatile("movq (%1, %2, 8),%0"
                                 : "=r"(tmpVal)
                                 : "r"(data), "r"((long)l));
                }
                auto accessTime = clock::now();
                for (auto l = 0; l < vec_size; l++) {
                    asm volatile("movq (%1, %2, 8),%0"
                                 : "=r"(tmpVal)
                                 : "r"(data), "r"((long)l));
                }
                auto accessFirst
                        = duration_cast<unit>(clock::now() - accessTime)
                                  .count();
                auto const msg_size = msg_sizes[msg_idx];
                auto msg = new ((std::align_val_t)32) char[msg_size];
                if (rank == 0) {
                    for (int l = 0; l < msg_size; l++) {
                        msg[l] = (char)(l * msg_idx * vec_idx);
                    }
                }
                auto sendTime = clock::now();
                func(msg, msg_size);
                auto sendElapsed
                        = duration_cast<unit>(clock::now() - sendTime).count();
                accessTime = clock::now();
                for (auto l = 0; l < vec_size; l++) {
                    asm volatile("movq (%1, %2, 8),%0"
                                 : "=r"(tmpVal)
                                 : "r"(data), "r"((long)l));
                }
                auto accessSecond
                        = duration_cast<unit>(clock::now() - accessTime)
                                  .count();
                fAccess[sample] = accessFirst;
                sAccess[sample] = accessSecond;
                sendTimes[sample] = sendElapsed;
                delete[] msg;
            }
            std::sort(fAccess.begin(), fAccess.end());
            std::sort(sAccess.begin(), sAccess.end());
            std::sort(sendTimes.begin(), sendTimes.end());
            auto fAccessMin = fAccess[0];
            auto sAccessMin = sAccess[0];
            auto fAccessMedian = median(fAccess);
            auto sAccessMedian = median(sAccess);

            auto accessSlowdownMin = (double)sAccessMin / (double)fAccessMin;
            // auto accessSlowdownMedian
            //         = (double)sAccessMedian / (double)fAccessMedian;

            auto sendTimeMin = sendTimes[0];
            // auto sendTimeMedian = median(sendTimes);

            fprintf(csvFile,
                    "%d" CSV_SEP "%d" CSV_SEP "%s" CSV_SEP "%ld" CSV_SEP
                    "%ld" CSV_SEP
                    "%"
                    "lf" CSV_SEP "%ld" CSV_SEP "%ld\n",
                    vec_sizes[vec_idx],
                    msg_sizes[msg_idx],
                    "ns",
                    fAccessMin,
                    sAccessMin,
                    accessSlowdownMin,
                    sAccessMin - fAccessMin,
                    sendTimeMin);
        }
    }
    fclose(csvFile);
}