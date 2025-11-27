#ifndef REVERSE_CUTHILL_MCKEE_H
#define REVERSE_CUTHILL_MCKEE_H

// Function to compute Reverse Cuthill-McKee ordering.
// Taken from sparsepak:
// https://people.sc.fsu.edu/~jburkardt/f77_src/sparsepak/sparsepak.html
// available under MIT license
// and modified to include it in HiGHS.
//

void genrcm(int node_num, int adj_num, int adj_row[], int adj[], int perm[]);

#endif