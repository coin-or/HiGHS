/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsTransformedLp.h
 * @brief LP transformations useful for cutting plane separation. This includes
 * bound substitution with simple and variable bounds, handling of slack
 * variables, flipping the complementation of integers.
 */

#ifndef MIP_HIGHS_TRANSFORMED_LP_H_
#define MIP_HIGHS_TRANSFORMED_LP_H_

#include <vector>

#include "HighsLpRelaxation.h"
#include "lp_data/HConst.h"
#include "mip/HighsImplications.h"
#include "util/HighsCDouble.h"
#include "util/HighsInt.h"
#include "util/HighsSparseVectorSum.h"

class HighsLpRelaxation;

/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
class HighsTransformedLp {
 private:
  const HighsLpRelaxation& lprelaxation;

  std::vector<std::pair<HighsInt, HighsImplications::VarBound>> bestVub;
  std::vector<std::pair<HighsInt, HighsImplications::VarBound>> bestVlb;
  std::vector<double> simpleLbDist;
  std::vector<double> simpleUbDist;
  std::vector<double> lbDist;
  std::vector<double> ubDist;
  std::vector<double> boundDist;
  enum class BoundType : uint8_t {
    kSimpleUb,
    kSimpleLb,
    kVariableUb,
    kVariableLb,
  };
  std::vector<BoundType> boundTypes;
  HighsSparseVectorSum vectorsum;

 public:
  HighsTransformedLp(const HighsLpRelaxation& lprelaxation,
                     HighsImplications& implications);

  double boundDistance(HighsInt col) const { return boundDist[col]; }

  // TODO: Should this simple be calculated when we initialise the transLp?
  double getFracVbEstimate(HighsInt col) const {
    HighsInt vbCol = -1;
    double vbCoef = 0.0;
    if (ubDist[col] <= lbDist[col] && bestVlb[col].first != -1) {
      vbCol = bestVlb[col].first;
      vbCoef = bestVlb[col].second.coef;
    } else if (lbDist[col] <= ubDist[col] && bestVub[col].first != -1) {
      vbCol = bestVub[col].first;
      vbCoef = bestVub[col].second.coef;
    } else if (bestVlb[col].first != -1) {
      vbCol = bestVlb[col].first;
      vbCoef = bestVlb[col].second.coef;
    } else if (bestVub[col].first != -1) {
      vbCol = bestVub[col].first;
      vbCoef = bestVub[col].second.coef;
    }
    if (vbCol == -1) return 0;
    // TODO: This differs from the SCIP definition? Their frac can be > 1?
    double val = vbCoef * lprelaxation.solutionValue(col);
    double frac = std::max(val - std::floor(val), 0.0);
    return std::min(frac, 1 - frac);
  }

  bool transform(std::vector<double>& vals, std::vector<double>& upper,
                 std::vector<double>& solval, std::vector<HighsInt>& inds,
                 double& rhs, bool& integralPositive, bool preferVbds = false);

  bool untransform(std::vector<double>& vals, std::vector<HighsInt>& inds,
                   double& rhs, bool integral = false);
};

#endif
