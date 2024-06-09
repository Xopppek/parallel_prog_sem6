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
        u[i][0] = sin(M_PI*i/N);
        u[i][N-1] = exp(-i/N)*sin(M_PI*i/N);
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

    int num_threads = 1;
    printf("num_threads = ");
    scanf("%d", &num_threads);
    omp_set_num_threads(num_threads);

    initialization(u);
    jacobi_method(u, u_temp);

    FILE* file = fopen("output.txt", "w");
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++)
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