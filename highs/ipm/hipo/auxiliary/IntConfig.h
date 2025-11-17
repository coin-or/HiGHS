#ifndef HIPO_INT_CONFIG_H
#define HIPO_INT_CONFIG_H

#include "lp_data/HConst.h"
#include "util/HighsInt.h"

namespace hipo {

// Generic integer type from HiGHS
typedef HighsInt Int;

// Integer type for indices of matrices in HiPO and factorisation
typedef int64_t Int64;

// The matrix (AS or NE) size must fit into Int.
// Type Int64 is used only for the nonzeros of the matrix and during the
// factorisation.
//
// For NE, AS, factorisations:
// - ptr and rows are std::vector<Index>.
// - ptr can be accessed with Int, rows and val must be accessed with Index.
// - rows[i] can be stored as Int, ptr[i] must be stored as Index.
//

}  // namespace hipo

#endif