#ifndef FACTORHIGHS_DGEMM_PARALLEL_H
#define FACTORHIGHS_DGEMM_PARALLEL_H

#include "DataCollector.h"
#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

// parallelise dgemm for use within factorisation
// Performs Q <- Q - R P^T in hybrid format.
// Parallelised over the rows of R and Q.
class dgemmParalleliser {
  const double* P_;
  const double* R_;
  double* Q_;
  const Int64 col_;
  const Int64 jb_;
  DataCollector& data_;

 public:
  dgemmParalleliser(const double* P, const double* R, double* Q, Int64 col,
                    Int64 jb, DataCollector& data);

  void run(Int64 start, Int64 end, double beta) const;
};

void dgemmParallel(const double* P, const double* R, double* Q, Int64 col, Int64 jb,
                   Int64 row, Int64 nb, double beta, DataCollector& data);

}  // namespace hipo

#endif