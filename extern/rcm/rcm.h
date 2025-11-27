#ifndef REVERSE_CUTHILL_MCKEE_H
#define REVERSE_CUTHILL_MCKEE_H

#include "util/HighsInt.h"

// Function to compute Reverse Cuthill-McKee ordering.
// Taken from sparsepak:
// https://people.sc.fsu.edu/~jburkardt/f77_src/sparsepak/sparsepak.html
// available under MIT license.
// Changes:
// - Type int substituted with HighsInt
// - Added return codes for errors.

HighsInt genrcm(HighsInt node_num, HighsInt adj_num, HighsInt adj_row[],
                HighsInt adj[], HighsInt perm[]);

#endif