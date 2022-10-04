#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <stdlib.h>
#include <vector>

int main()
{
    using clock = std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using unit = std::chrono::microseconds;
    using precise_unit = std::chrono::nanoseconds;

    setenv("MPI_SIZE", "2", 1);
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int vec_sizes[]
            = {4096, 102400, 204800, 409600, 921600, 3145728, 6000000};
    constexpr int msg_sizes[] = {4096, 102400, 204800, 409600, 921600, 3145728};

    constexpr int nsamples = 30;
    std::array<std::pair<double, long>, nsamples> samples;

    if (rank == 0) {
        for (int vec_idx = 0; vec_idx < (int)std::size(vec_sizes); vec_idx++) {
            auto const vec_size = vec_sizes[vec_idx];
            auto vec = std::vector<char>(vec_size);
            for (int l = 0; l < (int)vec.size(); l++) {
                vec[l] = l;
            }

            for (int msg_idx = 0; msg_idx < (int)std::size(msg_sizes);
                 msg_idx++) {
                for (int sample = 0; sample < nsamples; sample++) {
                    auto accessTime = clock::now();
                    for (int l = 0; l < (int)vec.size(); l++) {
                        vec[l] = l;
                    }
                    auto accessFirst = duration_cast<precise_unit>(
                                               clock::now() - accessTime)
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
                    auto accessSecond = duration_cast<precise_unit>(
                                                clock::now() - accessTime)
                                                .count();
                    //                    printf("%d | %d: access slowdown %lf,
                    //                    send time: %ld\n",
                    //                           vec_size,
                    //                           msg_size,
                    //                           (double)accessFirst /
                    //                           (double)accessSecond,
                    //                           sendElapsed);
                    samples[sample].first
                            = (double)accessSecond / (double)accessFirst;
                    samples[sample].second = sendElapsed;
                    delete[] msg;
                }
                std::sort(samples.begin(), samples.end());
                if (nsamples % 2 != 0) {
                    auto [accessTime, sendTime] = samples[nsamples / 2 + 1];
                    printf("%d | %d: access slowdown: %lf, send time: %ld\n",
                           vec_size,
                           msg_sizes[msg_idx],
                           accessTime,
                           sendTime);
                } else {
                    auto accessTime = (samples[nsamples / 2].first
                                       + samples[nsamples / 2 + 1].first)
                            / 2;
                    auto sendTime = (samples[nsamples / 2].second
                                     + samples[nsamples / 2 + 1].second)
                            / 2;
                    printf("%d | %d: access slowdown: %lf, send time: %ld\n",
                           vec_size,
                           msg_sizes[msg_idx],
                           accessTime,
                           sendTime);
                }
            }
        }
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
