#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[]){
    
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_even;
    if (size % 2 == 0)
        n_even = size / 2;
    else
        n_even = size / 2 + 1;

    int even_ranks[n_even];
    for (int i = 0; i < n_even; i++)
        even_ranks[i] = 2*i;

    int value;
    if (rank == 0)
        value = 0;
    else
        value = 1;
    
    MPI_Group group_world, group_even, group_odd;
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPI_Group_incl(group_world, n_even, even_ranks, &group_even);
    MPI_Group_excl(group_world, n_even, even_ranks, &group_odd);

    MPI_Comm MPI_COMM_EVEN, MPI_COMM_ODD;
    MPI_Comm_create(MPI_COMM_WORLD, group_even, &MPI_COMM_EVEN);
    MPI_Comm_create(MPI_COMM_WORLD, group_odd, &MPI_COMM_ODD);
    
    if (MPI_COMM_EVEN != MPI_COMM_NULL)
        MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_EVEN);
    

    if (MPI_COMM_ODD != MPI_COMM_NULL){
        MPI_Bcast(&value, 1, MPI_INT, 1, MPI_COMM_ODD);
    }

    printf("In process %d we have %d as value\n", rank, value);

    if (MPI_COMM_EVEN != MPI_COMM_NULL)
        MPI_Comm_free(&MPI_COMM_EVEN);
    else if (MPI_COMM_ODD != MPI_COMM_NULL)
        MPI_Comm_free(&MPI_COMM_ODD);

    MPI_Group_free(&group_world);
    MPI_Group_free(&group_even);
    MPI_Group_free(&group_odd);

    MPI_Finalize();

    return 0;
}