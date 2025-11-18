#ifndef HIPO_INT_CONFIG_H
#define HIPO_INT_CONFIG_H

#include "lp_data/HConst.h"
#include "util/HighsInt.h"

namespace hipo {

// Generic integer type from HiGHS
typedef HighsInt Int;

// Integer type for factorisation
typedef int64_t Int64;

// The matrix (AS or NE) is formed using Int, so it must have fewer than
// kHighsIInf nonzero entries. Metis works with the same type as Int, so it must
// be compiled accordingly.
//
// The factorisation uses Int64 everywhere, apart from where it interfaces with
// the matrix stored using Int.
// BLAS is 32-bit, so the vectors used by BLAS must be addressable with 32-bit
// integers.
//

}  // namespace hipo

#endif