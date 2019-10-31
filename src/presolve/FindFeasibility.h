#ifndef PRESOVLE_FINDFEASIBILITY_H_
#define PRESOVLE_FINDFEASIBILITY_H_

#include "lp_data/HighsLp.h"
#include "lp_data/HighsStatus.h"
#include "util/HighsTimer.h"

double getQuadraticObjective(const std::vector<double> cost,
                             const std::vector<double>& x,
                             const std::vector<double>& r, const double mu,
                             const std::vector<double> lambda);

enum class MinimizationType {
  kComponentWise,
  kComponentWisePenalty,
  kComponentWiseAdmm,
  kComponentWiseBreakpoints,
  kExact,
  kExactPenalty,
  kExactAdmm
};

HighsStatus runFeasibility(const HighsLp& lp, HighsSolution& solution,
                           const MinimizationType type,
                           const double initial_weight, HighsTimer& timer);

#endif