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
    
    MPI_Group group_world, group_even, group_odd_excl, group_odd_diff;
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPI_Group_incl(group_world, n_even, even_ranks, &group_even);

    MPI_Group_excl(group_world, n_even, even_ranks, &group_odd_excl);

    MPI_Group_difference(group_world, group_even, &group_odd_diff);
    
    int groups_ident_status;
    MPI_Group_compare(group_odd_excl, group_odd_diff, &groups_ident_status);

    if (groups_ident_status == MPI_IDENT)
        printf("From process %d view groups are identical\n", rank);
    else
        printf("From process %d view groups aren't identical\n", rank);

    MPI_Group_free(&group_world);
    MPI_Group_free(&group_even);
    MPI_Group_free(&group_odd_diff);
    MPI_Group_free(&group_odd_excl);

    MPI_Finalize();

    return 0;
}