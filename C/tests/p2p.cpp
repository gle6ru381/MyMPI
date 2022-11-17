#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char const* msg = "Hello world!";
    int len = strlen(msg);

    if (rank == 0) {
        MPI_Send(msg, len + 1, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        usleep(100000);
    } else {
        char buff[100];
        MPI_Status stat;
        MPI_Recv(buff, len + 1, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stat);
        fprintf(stderr, "Message: %s\n", buff);
    }
    MPI_Finalize();
}