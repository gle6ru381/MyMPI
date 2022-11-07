#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <stdlib.h>
#include <vector>

template <typename T, size_t N>
T median(std::array<T, N> const& arr)
{
    return arr.size() % 2 == 0
            ? (arr[arr.size() / 2] + arr[arr.size() / 2 + 1]) / 2
            : arr[arr.size() / 2 + 1];
}

int main()
{
    using clock = std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using unit = std::chrono::nanoseconds;

    setenv("MPI_SIZE", "2", 1);
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int vec_sizes[]
            = {4096, 125952, 204800, 458752, 921600, 2097152, 3932160};
    constexpr int msg_sizes[]
            = {4096, 125952, 204800, 458752, 921600, 2097152, 3932160};

    constexpr int nsamples = 500;
    std::array<long, nsamples> fAccess;
    std::array<long, nsamples> sAccess;
    std::array<long, nsamples> sendTimes;

    if (rank == 0) {
        auto csvFile = fopen("memcpy.csv", "w");
        fprintf(csvFile,
                "Vector size,Message size,unit,First access min,First access "
                "median,Second access "
                "min,"
                "Second access median,Access "
                "slowdown min,Access slowdown "
                "median,Send time min,Send "
                "time median\n");
        for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
            auto const vec_size = vec_sizes[vec_idx];
            auto vec = std::vector<char>(vec_size);
            for (int l = 0; l < (int)vec.size(); l++) {
                vec[l] = l;
            }

            for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes);
                 msg_idx++) {
                for (int sample = 0; sample < nsamples; sample++) {
                    for (int l = 0; l < (int)vec.size(); l++) {
                        vec[l] = l;
                    }
                    auto accessTime = clock::now();
                    for (int l = 0; l < (int)vec.size(); l++) {
                        vec[l] = l;
                    }
                    auto accessFirst
                            = duration_cast<unit>(clock::now() - accessTime)
                                      .count();
                    auto const msg_size = msg_sizes[msg_idx];
                    auto msg = new ((std::align_val_t)32) char[msg_size];
                    for (int l = 0; l < msg_size; l++) {
                        msg[l] = (char)(l * msg_idx * vec_idx);
                    }
                    auto sendTime = clock::now();
                    MPI_Send(msg, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    auto sendElapsed
                            = duration_cast<unit>(clock::now() - sendTime)
                                      .count();
                    accessTime = clock::now();
                    for (int l = 0; l < (int)vec.size(); l++) {
                        vec[l] = l;
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
                //                auto fAccessMax = fAccess[nsamples - 1];
                auto sAccessMin = sAccess[0];
                //                auto sAccessMax = sAccess[nsamples - 1];
                //                assert(fAccessMin <= fAccessMax);
                //                assert(sAccessMin <= sAccessMin);
                //                auto fAccessAvg
                //                        = std::accumulate(fAccess.begin(),
                //                        fAccess.end(), 0l) / nsamples;
                //                auto sAccessAvg
                //                        = std::accumulate(sAccess.begin(),
                //                        sAccess.end(), 0l) / nsamples;
                auto fAccessMedian = median(fAccess);
                auto sAccessMedian = median(sAccess);

                auto accessSlowdownMin
                        = (double)sAccessMin / (double)fAccessMin;
                //                auto accessSlowdownMax
                //                        = (double)sAccessMax /
                //                        (double)fAccessMax;
                //                auto accessSlowdownAvg
                //                        = (double)sAccessAvg /
                //                        (double)fAccessAvg;
                auto accessSlowdownMedian
                        = (double)sAccessMedian / (double)fAccessMedian;

                auto sendTimeMin = sendTimes[0];
                //                auto sendTimeMax = sendTimes[nsamples - 1];
                //                auto sendTimeAvg
                //                        = std::accumulate(
                //                                  sendTimes.begin(),
                //                                  sendTimes.end(), 0l)
                //                        / nsamples;
                auto sendTimeMedian = median(sendTimes);

                fprintf(csvFile,
                        "%d,%d,%s,%ld,%ld,%ld,%ld,%lf,%"
                        "lf,"
                        "%ld,%ld\n",
                        vec_size,
                        msg_sizes[msg_idx],
                        "ns",
                        fAccessMin,
                        fAccessMedian,
                        sAccessMin,
                        sAccessMedian,
                        accessSlowdownMin,
                        accessSlowdownMedian,
                        sendTimeMin,
                        sendTimeMedian);
                //                if (nsamples % 2 != 0) {
                //                    auto [accessTime, sendTime] =
                //                    samples[nsamples / 2 + 1]; printf("%d |
                //                    %d: access slowdown: %lf, send time:
                //                    %ld\n",
                //                           vec_size,
                //                           msg_sizes[msg_idx],
                //                           accessTime,
                //                           sendTime);
                //                    fprintf(csvFile,
                //                            "%d,%d,%lf,%ld\n",
                //                            vec_size,
                //                            msg_sizes[msg_idx],
                //                            accessTime,
                //                            sendTime);
                //                } else {
                //                    auto accessTime = (samples[nsamples /
                //                    2].first
                //                                       + samples[nsamples / 2
                //                                       + 1].first)
                //                            / 2;
                //                    auto sendTime = (samples[nsamples /
                //                    2].second
                //                                     + samples[nsamples / 2 +
                //                                     1].second)
                //                            / 2;
                //                    printf("%d | %d: access slowdown: %lf,
                //                    send time: %ld\n",
                //                           vec_size,
                //                           msg_sizes[msg_idx],
                //                           accessTime,
                //                           sendTime);
                //                    fprintf(csvFile,
                //                            "%d,%d,%lf,%ld\n",
                //                            vec_size,
                //                            msg_sizes[msg_idx],
                //                            accessTime,
                //                            sendTime);
                //                }
            }
        }
        fclose(csvFile);
    } else {
        for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
            for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes);
                 msg_idx++) {
                for (auto sample = 0; sample < nsamples; sample++) {
                    auto msg_size = msg_sizes[msg_idx];
                    auto msg = new ((std::align_val_t)32) char[msg_size];
                    MPI_Status stat;
                    for (int l = 0; l < msg_size; l++) {
                        msg[l] = 0;
                    }
                    MPI_Recv(
                            msg,
                            msg_size,
                            MPI_BYTE,
                            0,
                            0,
                            MPI_COMM_WORLD,
                            &stat);
                    for (int l = 0; l < msg_size; l++) {
                        if (msg[l] != (char)(l * msg_idx * vec_idx)) {
                            assert(false);
                        }
                    }
                    delete[] msg;
                }
            }
        }
    }
    std::cerr << rank << " finalize\n";
    MPI_Finalize();
}
