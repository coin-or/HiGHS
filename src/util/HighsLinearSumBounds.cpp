/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsLinearSumBounds.cpp
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#include "util/HighsLinearSumBounds.h"

#include <algorithm>  // For std::max

void HighsLinearSumBounds::add(int sum, int var, double coefficient) {
  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);
  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    // coefficient is positive, therefore variable lower contributes to sum
    // lower bound
    if (vLower == -HIGHS_CONST_INF)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vLower * coefficient;

    if (vUpper == HIGHS_CONST_INF)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vUpper * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varLower[var] * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varUpper[var] * coefficient;
  } else {
    // coefficient is negative, therefore variable upper contributes to sum
    // lower bound
    if (vUpper == HIGHS_CONST_INF)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vUpper * coefficient;

    if (vLower == -HIGHS_CONST_INF)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vLower * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varUpper[var] * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::remove(int sum, int var, double coefficient) {
  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);
  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    // coefficient is positive, therefore variable lower contributes to sum
    // lower bound
    if (vLower == -HIGHS_CONST_INF)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= vLower * coefficient;

    if (vUpper == HIGHS_CONST_INF)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= vUpper * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= varLower[var] * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= varUpper[var] * coefficient;
  } else {
    // coefficient is negative, therefore variable upper contributes to sum
    // lower bound
    if (vUpper == HIGHS_CONST_INF)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= vUpper * coefficient;

    if (vLower == -HIGHS_CONST_INF)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= vLower * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= varUpper[var] * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedVarUpper(int sum, int var, double coefficient,
                                           double oldVarUpper) {
  double oldVUpper = implVarUpperSource[var] == sum
                         ? oldVarUpper
                         : std::min(implVarUpper[var], oldVarUpper);

  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (coefficient > 0) {
    if (vUpper != oldVUpper) {
      if (oldVUpper == HIGHS_CONST_INF)
        numInfSumUpper[sum] -= 1;
      else
        sumUpper[sum] -= oldVUpper * coefficient;

      if (vUpper == HIGHS_CONST_INF)
        numInfSumUpper[sum] += 1;
      else
        sumUpper[sum] += vUpper * coefficient;
    }
    if (oldVarUpper == HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= oldVarUpper * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varUpper[var] * coefficient;
  } else {
    if (vUpper != oldVUpper) {
      if (oldVUpper == HIGHS_CONST_INF)
        numInfSumLower[sum] -= 1;
      else
        sumLower[sum] -= oldVUpper * coefficient;

      if (vUpper == HIGHS_CONST_INF)
        numInfSumLower[sum] += 1;
      else
        sumLower[sum] += vUpper * coefficient;
    }
    if (oldVarUpper == HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= oldVarUpper * coefficient;

    if (varUpper[var] == HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varUpper[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedVarLower(int sum, int var, double coefficient,
                                           double oldVarLower) {
  double oldVLower = implVarLowerSource[var] == sum
                         ? oldVarLower
                         : std::max(implVarLower[var], oldVarLower);

  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);

  if (coefficient > 0) {
    if (vLower != oldVLower) {
      if (oldVLower == -HIGHS_CONST_INF)
        numInfSumLower[sum] -= 1;
      else
        sumLower[sum] -= oldVLower * coefficient;

      if (vLower == -HIGHS_CONST_INF)
        numInfSumLower[sum] += 1;
      else
        sumLower[sum] += vLower * coefficient;
    }

    if (oldVarLower == -HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] -= 1;
    else
      sumLowerOrig[sum] -= oldVarLower * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumLowerOrig[sum] += 1;
    else
      sumLowerOrig[sum] += varLower[var] * coefficient;

  } else {
    if (vLower != oldVLower) {
      if (oldVLower == -HIGHS_CONST_INF)
        numInfSumUpper[sum] -= 1;
      else
        sumUpper[sum] -= oldVLower * coefficient;

      if (vLower == -HIGHS_CONST_INF)
        numInfSumUpper[sum] += 1;
      else
        sumUpper[sum] += vLower * coefficient;
    }
    if (oldVarLower == -HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] -= 1;
    else
      sumUpperOrig[sum] -= oldVarLower * coefficient;

    if (varLower[var] == -HIGHS_CONST_INF)
      numInfSumUpperOrig[sum] += 1;
    else
      sumUpperOrig[sum] += varLower[var] * coefficient;
  }
}

void HighsLinearSumBounds::updatedImplVarUpper(int sum, int var,
                                               double coefficient,
                                               double oldImplVarUpper,
                                               int oldImplVarUpperSource) {
  double oldVUpper = oldImplVarUpperSource == sum
                         ? varUpper[var]
                         : std::min(oldImplVarUpper, varUpper[var]);

  double vUpper = implVarUpperSource[var] == sum
                      ? varUpper[var]
                      : std::min(implVarUpper[var], varUpper[var]);

  if (vUpper == oldVUpper) return;

  if (coefficient > 0) {
    if (oldVUpper == HIGHS_CONST_INF)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= oldVUpper * coefficient;

    if (vUpper == HIGHS_CONST_INF)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vUpper * coefficient;
  } else {
    if (oldVUpper == HIGHS_CONST_INF)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= oldVUpper * coefficient;

    if (vUpper == HIGHS_CONST_INF)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vUpper * coefficient;
  }
}

void HighsLinearSumBounds::updatedImplVarLower(int sum, int var,
                                               double coefficient,
                                               double oldImplVarLower,
                                               int oldImplVarLowerSource) {
  double oldVLower = oldImplVarLowerSource == sum
                         ? varLower[var]
                         : std::max(oldImplVarLower, varLower[var]);

  double vLower = implVarLowerSource[var] == sum
                      ? varLower[var]
                      : std::max(implVarLower[var], varLower[var]);

  if (vLower == oldVLower) return;

  if (coefficient > 0) {
    if (oldVLower == -HIGHS_CONST_INF)
      numInfSumLower[sum] -= 1;
    else
      sumLower[sum] -= oldVLower * coefficient;

    if (vLower == -HIGHS_CONST_INF)
      numInfSumLower[sum] += 1;
    else
      sumLower[sum] += vLower * coefficient;

  } else {
    if (oldVLower == -HIGHS_CONST_INF)
      numInfSumUpper[sum] -= 1;
    else
      sumUpper[sum] -= oldVLower * coefficient;

    if (vLower == -HIGHS_CONST_INF)
      numInfSumUpper[sum] += 1;
    else
      sumUpper[sum] += vLower * coefficient;
  }
}

double HighsLinearSumBounds::getResidualSumLower(int sum, int var,
                                                 double coefficient) const {
  switch (numInfSumLower[sum]) {
    case 0:
      if (coefficient > 0) {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return double(sumLower[sum] - vLower * coefficient);
      } else {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return double(sumLower[sum] - vUpper * coefficient);
      }
      break;
    case 1:
      if (coefficient > 0) {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return vLower == -HIGHS_CONST_INF ? double(sumLower[sum])
                                          : -HIGHS_CONST_INF;
      } else {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return vUpper == HIGHS_CONST_INF ? double(sumLower[sum])
                                         : -HIGHS_CONST_INF;
      }
      break;
    default:
      return -HIGHS_CONST_INF;
  }
}

double HighsLinearSumBounds::getResidualSumUpper(int sum, int var,
                                                 double coefficient) const {
  switch (numInfSumUpper[sum]) {
    case 0:
      if (coefficient > 0) {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return double(sumUpper[sum] - vUpper * coefficient);
      } else {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return double(sumUpper[sum] - vLower * coefficient);
      }
      break;
    case 1:
      if (coefficient > 0) {
        double vUpper = implVarUpperSource[var] == sum
                            ? varUpper[var]
                            : std::min(implVarUpper[var], varUpper[var]);
        return vUpper == HIGHS_CONST_INF ? double(sumUpper[sum])
                                         : HIGHS_CONST_INF;
      } else {
        double vLower = implVarLowerSource[var] == sum
                            ? varLower[var]
                            : std::max(implVarLower[var], varLower[var]);
        return vLower == -HIGHS_CONST_INF ? double(sumUpper[sum])
                                          : HIGHS_CONST_INF;
      }
      break;
    default:
      return HIGHS_CONST_INF;
  }
}

double HighsLinearSumBounds::getResidualSumLowerOrig(int sum, int var,
                                                     double coefficient) const {
  switch (numInfSumLowerOrig[sum]) {
    case 0:
      if (coefficient > 0)
        return double(sumLowerOrig[sum] - varLower[var] * coefficient);
      else
        return double(sumLowerOrig[sum] - varUpper[var] * coefficient);
      break;
    case 1:
      if (coefficient > 0)
        return varLower[var] == -HIGHS_CONST_INF ? double(sumLowerOrig[sum])
                                                 : -HIGHS_CONST_INF;
      else
        return varUpper[var] == HIGHS_CONST_INF ? double(sumLowerOrig[sum])
                                                : -HIGHS_CONST_INF;
      break;
    default:
      return -HIGHS_CONST_INF;
  }
}

double HighsLinearSumBounds::getResidualSumUpperOrig(int sum, int var,
                                                     double coefficient) const {
  switch (numInfSumUpperOrig[sum]) {
    case 0:
      if (coefficient > 0)
        return double(sumUpperOrig[sum] - varUpper[var] * coefficient);
      else
        return double(sumUpperOrig[sum] - varLower[var] * coefficient);
      break;
    case 1:
      if (coefficient > 0)
        return varUpper[var] == HIGHS_CONST_INF ? double(sumUpperOrig[sum])
                                                : HIGHS_CONST_INF;
      else
        return varLower[var] == -HIGHS_CONST_INF ? double(sumUpperOrig[sum])
                                                 : HIGHS_CONST_INF;
      break;
    default:
      return HIGHS_CONST_INF;
  }
}

void HighsLinearSumBounds::shrink(const std::vector<int>& newIndices,
                                  int newSize) {
  int oldNumInds = newIndices.size();
  for (int i = 0; i != oldNumInds; ++i) {
    if (newIndices[i] != -1) {
      sumLower[newIndices[i]] = sumLower[i];
      sumUpper[newIndices[i]] = sumUpper[i];
      numInfSumLower[newIndices[i]] = numInfSumLower[i];
      numInfSumUpper[newIndices[i]] = numInfSumUpper[i];
      sumLowerOrig[newIndices[i]] = sumLowerOrig[i];
      sumUpperOrig[newIndices[i]] = sumUpperOrig[i];
      numInfSumLowerOrig[newIndices[i]] = numInfSumLowerOrig[i];
      numInfSumUpperOrig[newIndices[i]] = numInfSumUpperOrig[i];
    }
  }

  sumLower.resize(newSize);
  sumUpper.resize(newSize);
  numInfSumLower.resize(newSize);
  numInfSumUpper.resize(newSize);
  sumLowerOrig.resize(newSize);
  sumUpperOrig.resize(newSize);
  numInfSumLowerOrig.resize(newSize);
  numInfSumUpperOrig.resize(newSize);
}
