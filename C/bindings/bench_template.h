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

#ifndef NEED_HEADER
#define NEED_HEADER 0
#endif

static constexpr int nsamples = 200;
static constexpr int vec_sizes[] = {15'360, 130'048, 10'480'640, 20'961'280};
static constexpr int msg_sizes[] = {15'360, 130'048, 10'480'640, 20'961'280};

constexpr int bench_max_size()
{
    return vec_sizes[std::size(vec_sizes) - 1] * 8;
}

template <typename T, size_t N>
T median(std::array<T, N> const& arr)
{
    return arr.size() % 2 == 0
            ? (arr[arr.size() / 2] + arr[arr.size() / 2 + 1]) / 2
            : arr[arr.size() / 2 + 1];
}

template <typename T, size_t N>
T quantile_25(std::array<T, N> const& arr)
{
    auto half = arr.size() / 2;
    return half % 2 == 0 ? (arr[half / 2] + arr[half / 2 + 1]) / 2
                         : arr[half / 2 + 1];
}

template <typename T, size_t N>
T quantile_75(std::array<T, N> const& arr)
{
    auto half = arr.size() / 2;
    return half % 2 == 0 ? (arr[half + half / 2] + arr[half + half / 2 + 1]) / 2
                         : arr[half + half / 2 + 1];
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

    std::array<long, nsamples> fAccess;
    std::array<long, nsamples> sAccess;
    std::array<long, nsamples> sendTimes;

    auto csvFile = fopen(fileName.c_str(), "w");
#if NEED_HEADER
    fprintf(csvFile,
            "Size" CSV_SEP "unit" CSV_SEP
            "First access "
            "min" CSV_SEP "Second access min" CSV_SEP
            "Access slowdown "
            "min" CSV_SEP "Access diff" CSV_SEP
            "Send "
            "time min\n");
#endif

    std::cerr << "Run bench\n";

    for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
        auto const vec_size = vec_sizes[vec_idx] / 8;
        auto vec = std::vector<long>(vec_size);
        auto const msg_idx = vec_idx;

        //        for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes);
        //        msg_idx++) {
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
                    = duration_cast<unit>(clock::now() - accessTime).count();
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
                    = duration_cast<unit>(clock::now() - accessTime).count();
            fAccess[sample] = accessFirst;
            sAccess[sample] = accessSecond;
            sendTimes[sample] = sendElapsed;
            delete[] msg;
        }
        std::array<double, nsamples> accessSlowdown;
        for (int i = 0; i < nsamples; i++) {
            accessSlowdown[i] = (double)sAccess[i] / (double)fAccess[i];
        }
        std::sort(accessSlowdown.begin(), accessSlowdown.end());
        std::sort(sendTimes.begin(), sendTimes.end());

        auto accessSlowdownMin = accessSlowdown[0];
        auto accessSlowdownMedian = median(accessSlowdown);
        auto accessSlowdownMax = accessSlowdown[nsamples - 1];
        auto accessSlowdownQuantile25 = quantile_25(accessSlowdown);
        auto accessSlowdownQuantile75 = quantile_75(accessSlowdown);

        auto sendTimeMin = sendTimes[0];
        auto sendTimeMedian = median(sendTimes);
        auto sendTimeMax = sendTimes[nsamples - 1];
        auto sendTimeQuantile25 = quantile_25(sendTimes);
        auto sendTimeQuantile75 = quantile_75(sendTimes);

        fprintf(csvFile,
                "%d-%d" CSV_SEP "%s" CSV_SEP "%lf" CSV_SEP "%lf" CSV_SEP
                "%lf" CSV_SEP "%lf" CSV_SEP "%lf" CSV_SEP "%ld" CSV_SEP
                "%ld" CSV_SEP "%ld" CSV_SEP "%ld" CSV_SEP
                "%ld"
                "\n",
                vec_sizes[vec_idx],
                msg_sizes[msg_idx],
                "ns",
                accessSlowdownMin,
                accessSlowdownMedian,
                accessSlowdownMax,
                accessSlowdownQuantile25,
                accessSlowdownQuantile75,
                sendTimeMin,
                sendTimeMedian,
                sendTimeMax,
                sendTimeQuantile25,
                sendTimeQuantile75);
    }
    //    }
    fclose(csvFile);
}

template <typename Predicate>
void block_bench(char const* fName, Predicate func)
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

    std::array<long, nsamples> fAccess;
    std::array<long, nsamples> sAccess;
    std::array<long, nsamples> sendTimes;

    auto csvFile = fopen(fileName.c_str(), "w");
#if NEED_HEADER
    fprintf(csvFile,
            "Size" CSV_SEP "unit" CSV_SEP
            "First access "
            "min" CSV_SEP "Second access min" CSV_SEP
            "Access slowdown "
            "min" CSV_SEP "Access diff" CSV_SEP
            "Send "
            "time min\n");
#endif

    for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
        auto const vec_size = vec_sizes[vec_idx] / 8;
        auto const msg_idx = vec_idx;
        auto vec = std::vector<long>(vec_size);

        //        for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes);
        //        msg_idx++) {
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
                    = duration_cast<unit>(clock::now() - accessTime).count();
            auto const msg_size = msg_sizes[msg_idx];
            auto msg = new ((std::align_val_t)32) char[msg_size];
            auto step = msg_size / size;
            for (int l = step * rank; l < step * rank + step; l++) {
                msg[l] = (char)(l * msg_idx * vec_idx);
            }
            auto sendTime = clock::now();
            func(msg, msg_size, rank, size);
            auto sendElapsed
                    = duration_cast<unit>(clock::now() - sendTime).count();
            accessTime = clock::now();
            for (auto l = 0; l < vec_size; l++) {
                asm volatile("movq (%1, %2, 8),%0"
                             : "=r"(tmpVal)
                             : "r"(data), "r"((long)l));
            }
            auto accessSecond
                    = duration_cast<unit>(clock::now() - accessTime).count();
            fAccess[sample] = accessFirst;
            sAccess[sample] = accessSecond;
            sendTimes[sample] = sendElapsed;
            delete[] msg;
        }
        std::array<double, nsamples> accessSlowdown;
        for (int i = 0; i < nsamples; i++) {
            accessSlowdown[i] = (double)sAccess[i] / (double)fAccess[i];
        }
        std::sort(accessSlowdown.begin(), accessSlowdown.end());
        std::sort(sendTimes.begin(), sendTimes.end());

        auto accessSlowdownMin = accessSlowdown[0];
        auto accessSlowdownMedian = median(accessSlowdown);
        auto accessSlowdownMax = accessSlowdown[nsamples - 1];
        auto accessSlowdownQuantile25 = quantile_25(accessSlowdown);
        auto accessSlowdownQuantile75 = quantile_75(accessSlowdown);

        auto sendTimeMin = sendTimes[0];
        auto sendTimeMedian = median(sendTimes);
        auto sendTimeMax = sendTimes[nsamples - 1];
        auto sendTimeQuantile25 = quantile_25(sendTimes);
        auto sendTimeQuantile75 = quantile_75(sendTimes);

        fprintf(csvFile,
                "%d-%d" CSV_SEP "%s" CSV_SEP "%lf" CSV_SEP "%lf" CSV_SEP
                "%lf" CSV_SEP "%lf" CSV_SEP "%lf" CSV_SEP "%ld" CSV_SEP
                "%ld" CSV_SEP "%ld" CSV_SEP "%ld" CSV_SEP
                "%ld"
                "\n",
                vec_sizes[vec_idx],
                msg_sizes[msg_idx],
                "ns",
                accessSlowdownMin,
                accessSlowdownMedian,
                accessSlowdownMax,
                accessSlowdownQuantile25,
                accessSlowdownQuantile75,
                sendTimeMin,
                sendTimeMedian,
                sendTimeMax,
                sendTimeQuantile25,
                sendTimeQuantile75);
    }
    //    }
    fclose(csvFile);
}