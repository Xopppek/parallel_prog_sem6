#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include "omp.h"

#define N 100
#define MAX_ITERATIONS 10000

void initialization(double** u){

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            u[i][j] = 0.0;
    
    for (int i = 0; i < N; i++){  //up and bottom boundaries
        u[i][0] = sin(M_PI*i*1.0/(N-1));
        u[i][N-1] = exp(-1.0*i/(N-1))*sin(M_PI*i*1.0/(N-1));
    }
}

void jacobi_method(double** u, double** u_temp){

    int current_iteration = 0;

    while (current_iteration < MAX_ITERATIONS){

        #pragma omp parallel for
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                u_temp[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
            
        #pragma omp parallel for
        for (int i = 1; i < N-1; i++)
            for (int j = 1; j < N-1; j++)
                u[i][j] = u_temp[i][j];

        current_iteration++;
    }
}


int main(int argc, char* argv[]){

    double** u = (double**) malloc(N * sizeof(double*));
    double** u_temp = (double**) malloc(N * sizeof(double*));

    for (int i = 0; i < N; i++){
        u[i] = (double*) malloc(N * sizeof(double));
        u_temp[i] = (double*) malloc(N * sizeof(double));
    }

    double time1;
    for (int threads = 1; threads <= 12; threads++){
        double num_threads = threads;

        omp_set_num_threads(num_threads);

        double start_time = omp_get_wtime();

        initialization(u);
        jacobi_method(u, u_temp);

        double end_time = omp_get_wtime();
        double delta_time = end_time - start_time;

        if (threads == 1)
            time1 = delta_time;
        else
            printf("Ускорение для %d процессов %lf\n", threads, time1/delta_time);
    }
    
    FILE* file = fopen("output.txt", "w");
        for (int j = 0; j < N; j++){
            for (int i = 0; i < N; i++)
                fprintf(file, "%lf ", u[i][j]);
            fprintf(file, "\n");
        }
    fclose(file);

    for (int i = 0; i < N; i++){
        free(u[i]);
        free(u_temp[i]);
    }

    free(u);
    free(u_temp);

    return 0;
}