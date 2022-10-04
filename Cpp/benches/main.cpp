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

    setenv("MPI_SIZE", "2", 1);
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int vec_sizes[] = {4096, 102400, 204800, 409600, 921600, 3145728};
    constexpr int msg_sizes[] = {4096, 102400, 204800, 409600, 921600, 3145728};

    if (rank == 0) {
        for (int i = 0; i < (int)std::size(vec_sizes); i++) {
            auto const vec_size = vec_sizes[i];
            auto vec = std::vector<char>(vec_size);
            for (int l = 0; l < (int)vec.size(); l++) {
                vec[l] = l;
            }

            for (int j = 0; j < (int)std::size(msg_sizes); j++) {
                auto accessTime = clock::now();
                for (int l = 0; l < (int)vec.size(); l++) {
                    vec[l] = l;
                }
                auto accessFirst
                        = duration_cast<unit>(clock::now() - accessTime)
                                  .count();
                auto const msg_size = msg_sizes[j];
                auto msg = std::unique_ptr<char>(
                        new ((std::align_val_t)32) char[msg_size]);
                for (int l = 0; l < msg_size; l++) {
                    msg.get()[l] = l % 100;
                }
                auto sendTime = clock::now();
                MPI_Send(msg.get(), msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                auto sendElapsed
                        = duration_cast<unit>(clock::now() - sendTime).count();
                accessTime = clock::now();
                for (int l = 0; l < (int)vec.size(); l++) {
                    vec[l] = l;
                }
                auto accessSecond
                        = duration_cast<unit>(clock::now() - accessTime)
                                  .count();
                printf("%d | %d: access slowdown %lf, send time: %ld\n",
                       vec_size,
                       msg_size,
                       (double)accessFirst / (double)accessSecond,
                       sendElapsed);
                std::cerr << "Measurment\n";
            }
        }
    } else {
        for (int t = 0; t++ < (int)std::size(vec_sizes); t++) {
            for (int i = 0; i < (int)std::size(msg_sizes); i++) {
                auto msg_size = msg_sizes[i];
                auto msg = std::unique_ptr<char>(
                        new ((std::align_val_t)32) char[msg_size]);
                MPI_Status stat;
                for (int l = 0; l < msg_size; l++) {
                    msg.get()[i] = 0;
                }
                int ret = MPI_Recv(
                        msg.get(),
                        msg_size,
                        MPI_BYTE,
                        0,
                        0,
                        MPI_COMM_WORLD,
                        &stat);
                fprintf(stderr,
                        "%d | %d Receive end %d\n",
                        vec_sizes[i],
                        msg_size,
                        ret);
                for (int l = 0; l < msg_size; l++) {
                    if (msg.get()[l] != l % 100) {
                        fprintf(stderr,
                                "Error in pos: %d, expect: %d, true: %d\n",
                                l,
                                l % 100,
                                msg.get()[l]);
                        assert(false);
                    }
                }
            }
        }
    }
    std::cerr << rank << " finalize\n";
    MPI_Finalize();
}
