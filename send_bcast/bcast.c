#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>

#define ARRAY_SIZE 300000

int* get_rand_array(){
    srand(time(NULL));
    int* arr = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++){
        arr[i] = rand();
    }
    return arr;
}

int main(int argc, char* argv[]){
    int rank;
    double start_time, end_time;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* arr = get_rand_array();
    
    start_time = MPI_Wtime();
    MPI_Bcast(arr, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    MPI_Finalize();

    if (rank == 0){printf("%f\n", end_time - start_time);}

    return 0;
}