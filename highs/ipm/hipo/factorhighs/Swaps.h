#ifndef FACTORHIGHS_SWAPS_H
#define FACTORHIGHS_SWAPS_H

#include "DataCollector.h"
#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

void permuteWithSwaps(double* x, const Int64* swaps, Int64 n, bool reverse = false);

void swapCols(char uplo, Int64 n, double* A, Int64 lda, Int64 i, Int64 j, Int64* swaps,
              Int64* sign, DataCollector& data);

void applySwaps(const Int64* swaps, Int64 nrow, Int64 ncol, double* R,
                DataCollector& data);

}  // namespace hipo

#endif