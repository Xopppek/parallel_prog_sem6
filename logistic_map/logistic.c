#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "mpi.h"

#define R_POINTS_AMOUNT 10000
#define R_MIN 2.8
#define R_MAX 4.0
#define X0 0.49
#define ITERATIONS 1000000
#define EPSILON 1e-12

double* linspace(double start, double end, int points_amount){
    double* res = (double*)malloc(points_amount * sizeof(double));
    double step = (end - start)/(points_amount - 1);
    res[0] = start;
    for (int i = 1; i < points_amount; i++)
        res[i] = res[i-1] + step;
    return res;
}

double get_next_logistic(double x, double r){
    return r * x * (1-x);
}

bool is_equal(double a, double b, double e){
    return (a - b < e) && (b - a < e);
}

int count_unique(double* arr, int size, double e) {
    int count = 0;
    bool is_unique;

    for (int i = 0; i < size; ++i) {
        is_unique = true;
        for (int j = 0; j < i; ++j) {
            if (is_equal(arr[i], arr[j], e)) {
                is_unique = false;
                break;
            }
        }
        if (is_unique)
            count++;
    }
    return count;
}

int count_attractors(double r, double x0, double e){
    double xn = x0;
    for (int j = 0; j < ITERATIONS; j++)
        xn = get_next_logistic(xn, r);
    double last_50_values[50];
    for (int j = 0; j < 50; j++){
        xn = get_next_logistic(xn, r);
        last_50_values[j] = xn;
    }
    return count_unique(last_50_values, 50, e);
}

int main(int argc, char* argv[]){

    int nproc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int N_PER_PROCESS = R_POINTS_AMOUNT/nproc;
    double* r_values = NULL;
    double r_local[N_PER_PROCESS];
    double xn = X0;
    int attractors[R_POINTS_AMOUNT];
    int attractors_local[N_PER_PROCESS];

    if (rank == 0)
        r_values = linspace(R_MIN, R_MAX, R_POINTS_AMOUNT);

    MPI_Scatter(r_values, N_PER_PROCESS, MPI_DOUBLE, 
                r_local, N_PER_PROCESS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N_PER_PROCESS; i++)
        attractors_local[i] = count_attractors(r_local[i], X0, EPSILON);

    MPI_Gather(attractors_local, N_PER_PROCESS, MPI_INT, 
                attractors, N_PER_PROCESS, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0){
        for (int i = N_PER_PROCESS * nproc; i < R_POINTS_AMOUNT; i++)
            attractors[i] = count_attractors(r_values[i], X0, EPSILON);
        
        for (int i = 0; i < R_POINTS_AMOUNT; i++)
            printf("%f %d\n", r_values[i], attractors[i]);
    }

    free(r_values);
    MPI_Finalize();

    return 0;
}