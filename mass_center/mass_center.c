#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <time.h>
#include "mpi.h"

typedef struct {
    float x;
    float y;
} point;

void point_sum(void *in, void *inout, int *len, MPI_Datatype *dptr) {
    point *in_points = (point *)in;
    point *inout_points = (point *)inout;

    for (int i = 0; i < *len; i++) {
        inout_points[i].x += in_points[i].x;
        inout_points[i].y += in_points[i].y;
    }
}

point* get_points(int n) {
    point* points = (point*)malloc(n * sizeof(point));

    srand(time(NULL));

    for (int i = 0; i < n; ++i) {
        points[i].x = (float)((rand() % 201) - 100);
        points[i].y = (float)((rand() % 201) - 100);
    }

    return points;
}

float get_distance(point a, point b){
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int main(int argc, char* argv[]){

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N;

    if (rank == 0){
        printf("Enter amount of points: ");
        scanf("%d", &N);
    
        while (N%size != 0){
            printf("Please enter value that is devisible by %d: ", size);
            scanf("%d", &N);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(2, MPI_FLOAT, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    MPI_Op MPI_POINT_SUM;
    MPI_Op_create((MPI_User_function *) point_sum, 1, &MPI_POINT_SUM);  

    int N_PER_PROCESS = N/size;

    point* points = NULL;
    point local_points[N_PER_PROCESS];

    if (rank == 0){
        points = get_points(N);
        printf("Points: ");
        for (int i = 0; i < N; i++){
            printf("(%.2f, %.2f)", points[i].x, points[i].y);
            if (i != N-1)
                printf(", ");
            else
                printf(";\n");
        }
    }

    MPI_Scatter(points, N_PER_PROCESS, MPI_POINT, local_points, N_PER_PROCESS, MPI_POINT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    point local_r; local_r.x = 0; local_r.y = 0;

    for (int i = 0; i < N_PER_PROCESS; i++){
        local_r.x += local_points[i].x;
        local_r.y += local_points[i].y;
    }

    char filename[20];
    snprintf(filename, 20, "proc%d.txt", rank);

    FILE* file = fopen(filename, "w");
    
    for (int i = 0; i < N_PER_PROCESS; i++){
        fprintf(file, "%lf ", local_points[i].x);
    }

    fprintf(file, "\n");

    for (int i = 0; i < N_PER_PROCESS; i++){
        fprintf(file, "%lf ", local_points[i].y);
    }

    fclose(file);

    snprintf(filename, 20, "center_proc%d.txt", rank);

    FILE* file2 = fopen(filename, "w");
    
    fprintf(file2, "%lf ", local_r.x/N_PER_PROCESS);
    fprintf(file2, "\n");
    fprintf(file2, "%lf ", local_r.y/N_PER_PROCESS);

    fclose(file2);

    point center;

    MPI_Allreduce(&local_r, &center, 1, MPI_POINT, MPI_POINT_SUM, MPI_COMM_WORLD); 

    center.x /= N; center.y /= N;

    if (rank == 0){
        printf("Mass center: (%.2f, %.2f)\n", center.x, center.y);
        FILE* file = fopen("center.txt", "w");
        fprintf(file, "%lf\n", center.x);
        fprintf(file, "%lf", center.y);
        fclose(file);
    }
    
    for (int i = 0; i < N_PER_PROCESS; i++){
            printf("Distance between %d point and mass center is %.2f\n", i + N_PER_PROCESS*rank, get_distance(center, local_points[i]));
    }

    free(points);
    MPI_Op_free(&MPI_POINT_SUM);
    MPI_Type_free(&MPI_POINT);
    MPI_Finalize();

    return 0;
}