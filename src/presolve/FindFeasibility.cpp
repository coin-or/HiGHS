#include "FindFeasibility.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "io/HighsIO.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsLpUtils.h"
#include "presolve/ExactSubproblem.h"
#include "util/HighsUtils.h"

constexpr double kExitTolerance = 0.00000001;

bool isEqualityProblem(const HighsLp& lp) {
  for (int row = 0; row < lp.numRow_; row++)
    if (lp.rowLower_[row] != lp.rowUpper_[row]) return false;

  return true;
}

std::vector<double> getAtb(const HighsLp& lp) {
  assert(lp.rowUpper_ == lp.rowLower_);
  std::vector<double> atb(lp.numCol_, 0);
  for (int col = 0; col < lp.numCol_; col++) {
    for (int k = lp.Astart_[col]; k < lp.Astart_[col + 1]; k++) {
      const int row = lp.Aindex_[k];
      atb.at(col) += lp.Avalue_[k] * lp.rowUpper_[row];
    }
  }
  return atb;
}

std::vector<double> getAtLambda(const HighsLp& lp,
                                const std::vector<double> lambda) {
  std::vector<double> atl(lp.numCol_);
  for (int col = 0; col < lp.numCol_; col++) {
    for (int k = lp.Astart_[col]; k < lp.Astart_[col + 1]; k++) {
      const int row = lp.Aindex_[k];
      atl.at(col) += lp.Avalue_[k] * lambda[row];
    }
  }
  return atl;
}

double getQuadraticObjective(const std::vector<double> cost,
                             const std::vector<double>& x,
                             std::vector<double>& r, const double mu,
                             const std::vector<double> lambda) {
  assert(cost.size() == x.size());
  // c'x
  double quadratic = 0;
  for (int col = 0; col < (int)x.size(); col++) quadratic += cost[col] * x[col];

  // lambda'x
  for (int row = 0; row < (int)lambda.size(); row++) {
    quadratic += lambda[row] * r[row];
  }

  // 1/2mu r'r
  for (int row = 0; row < (int)lambda.size(); row++) {
    quadratic += (r[row] * r[row]) / (2 * mu);
  }

  return quadratic;
}

void printMinorIterationDetails(const double iteration, const double col,
                                const double old_value, const double update,
                                const double ctx, const std::vector<double>& r,
                                const double quadratic_objective) {
  double rnorm = getNorm2(r);
  std::cout << "iter " << iteration;
  std::cout << ", col " << col;
  std::cout << ", update " << update;
  std::cout << ", old_value " << old_value;
  std::cout << ", new_value " << old_value + update;
  std::cout << ", ctx " << ctx;
  std::cout << ", r " << rnorm;
  std::cout << ", quadratic_objective " << quadratic_objective;
  std::cout << std::endl;
}

class Quadratic {
 public:
  Quadratic(const HighsLp& lp, std::vector<double>& primal_values,
            ResidualFunctionType type)
      : lp_(lp), col_value_(primal_values) {
    update(type);
  }

  const std::vector<double>& getResidual() const { return residual_; }
  double getResidualNorm2() const { return residual_norm_2_; }
  double getObjective() const { return objective_; }

  void getSolution(HighsSolution& solution) const {
    solution.col_value = col_value_;
    solution.row_value = row_value_;

    // check what solution looks like
    double max = *std::max_element(col_value_.begin(), col_value_.end());
    double min = *std::min_element(col_value_.begin(), col_value_.end());

    HighsPrintMessage(ML_ALWAYS, "\n");
    HighsPrintMessage(ML_ALWAYS, "Solution max element: %4.3f\n", max);
    HighsPrintMessage(ML_ALWAYS, "Solution min element: %4.3f\n", min);
  }

  void minimize_by_component(const double mu, const std::vector<double>& lambda,
                             const ResidualFunctionType type);

  double minimize_component_quadratic_linearisation(
      const int col, const double mu, const std::vector<double>& lambda);
  double minimize_component_quadratic_piecewise(
      const int col, const double mu, const std::vector<double>& lambda);
  void minimize_exact_penalty(const double mu);
  void minimize_exact_with_lambda(const double mu,
                                  const std::vector<double>& lambda);

 private:
  const HighsLp& lp_;
  std::vector<double> col_value_;
  std::vector<double> row_value_;

  double objective_;
  double residual_norm_1_;
  double residual_norm_2_;
  std::vector<double> residual_;

  void updateObjective();
  void updateRowValue();
  void updateResidual(
      ResidualFunctionType quadratic_type = ResidualFunctionType::kLinearised);

  void update(
      ResidualFunctionType quadratic_type = ResidualFunctionType::kLinearised);

  // double calculateQuadraticValue(const double mu,
  //                                const std::vector<double> lambda,
  //                                ResidualFunctionType type);
  double findBreakpoints(const int col, const double mu,
                         const std::vector<double> lambda);
};

void Quadratic::update(ResidualFunctionType quadratic_type) {
  updateObjective();
  updateRowValue();
  updateResidual(quadratic_type);
}

void Quadratic::updateRowValue() {
  row_value_.clear();
  row_value_.assign(lp_.numRow_, 0);

  for (int col = 0; col < lp_.numCol_; col++) {
    for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
      int row = lp_.Aindex_[k];
      row_value_[row] += lp_.Avalue_[k] * col_value_[col];
    }
  }
}

void Quadratic::updateResidual(ResidualFunctionType quadratic_type) {
  residual_.clear();
  residual_.assign(lp_.numRow_, 0);
  residual_norm_1_ = 0;
  residual_norm_2_ = 0;

  if (quadratic_type == ResidualFunctionType::kLinearised) {
    for (int row = 0; row < lp_.numRow_; row++) {
      // for the moment assuming rowLower == rowUpper
      residual_[row] = lp_.rowUpper_[row] - row_value_[row];

      residual_norm_1_ += std::fabs(residual_[row]);
      residual_norm_2_ += residual_[row] * residual_[row];
    }

  } else if (quadratic_type == ResidualFunctionType::kPiecewise)
    for (int row = 0; row < lp_.numRow_; row++) {
      double value = 0;
      if (row_value_[row] <= lp_.rowLower_[row])
        value = lp_.rowLower_[row] - row_value_[row];
      else if (row_value_[row] >= lp_.rowUpper_[row])
        value = row_value_[row] - lp_.rowUpper_[row];

      residual_[row] = value;
      residual_norm_1_ += std::fabs(residual_[row]);
      residual_norm_2_ += residual_[row] * residual_[row];
    }

  residual_norm_2_ = std::sqrt(residual_norm_2_);
}

void Quadratic::updateObjective() {
  objective_ = 0;
  for (int col = 0; col < lp_.numCol_; col++)
    objective_ += lp_.colCost_[col] * col_value_[col];
}

HighsStatus initialize(const HighsLp& lp, HighsSolution& solution, double& mu,
                       std::vector<double>& lambda) {
  if (!isSolutionConsistent(lp, solution)) {
    // clear and resize solution.
    solution.col_value.clear();
    solution.col_dual.clear();
    solution.row_value.clear();
    solution.row_dual.clear();

    solution.col_value.resize(lp.numCol_);
  }

  for (int col = 0; col < lp.numCol_; col++) {
    if (lp.colLower_[col] <= 0 && lp.colUpper_[col] >= 0)
      solution.col_value[col] = 0;
    else if (lp.colLower_[col] > 0)
      solution.col_value[col] = lp.colLower_[col];
    else if (lp.colUpper_[col] < 0)
      solution.col_value[col] = lp.colUpper_[col];
    else {
      HighsLogMessage(HighsMessageType::ERROR,
                      "Error setting initial value for column %d", col);
      return HighsStatus::Error;
    }
  }

  lambda.resize(lp.numRow_);
  lambda.assign(lp.numRow_, 0);

  return HighsStatus::OK;
}

// Should only work with kLinearized type.
void Quadratic::minimize_exact_with_lambda(const double mu,
                                           const std::vector<double>& lambda) {
  double mu_penalty = 1.0 / mu;
  HighsLp lp = lp_;
  // Modify cost. See notebook ."lambda"
  // projected_gradient_c = c - 1/mu*(A'b) - A'\lambda
  // First part taken into consideration in projected gradient.

  std::vector<double> atb = getAtb(lp);
  std::vector<double> atlambda = getAtLambda(lp, lambda);
  for (int col = 0; col < (int)lp.colCost_.size(); col++)
    lp.colCost_[col] -= atlambda[col];

  solve_exact(lp, mu_penalty, col_value_);

  update();
}

void Quadratic::minimize_exact_penalty(const double mu) {
  double mu_penalty = 1.0 / mu;
  HighsLp lp = lp_;
  // Modify cost. See notebook ."no lambda"
  // projected_gradient_c = c - 1/mu*(A'b)
  // First part taken into consideration in projected gradient.

  solve_exact(lp, mu_penalty, col_value_);

  update();
}

double Quadratic::minimize_component_quadratic_linearisation(
    const int col, const double mu, const std::vector<double>& lambda) {
  // todo: see again when you refactor caclQV out of there
  // is this why ff became so slow? no, but the same call was done for each
  // component update below and that was the reason.
  // double current =
  //     calculateQuadraticValue(mu, lambda, ResidualFunctionType::kLinearised);

  // Minimize quadratic for column col.

  // Formulas for a and b when minimizing for x_j
  // a = (1/(2*mu)) * sum_i a_ij^2
  // b = -(1/(2*mu)) sum_i (2 * a_ij * (sum_{k!=j} a_ik * x_k - b_i)) + c_j (\)
      //     + sum_i a_ij * lambda_i
  // b / 2 = -(1/(2*mu)) sum_i (2 * a_ij
  double a = 0.0;
  double b = 0.0;

  for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
    int row = lp_.Aindex_[k];
    a += lp_.Avalue_[k] * lp_.Avalue_[k];
    // matlab but with b = b / 2
    double bracket = -residual_[row] - lp_.Avalue_[k] * col_value_[col];
    bracket += lambda[row];
    // clp minimizing for delta_x
    // double bracket_clp = - residual_[row];
    b += lp_.Avalue_[k] * bracket;
  }

  a = (0.5 / mu) * a;
  b = (0.5 / mu) * b + 0.5 * lp_.colCost_[col];

  double theta = -b / a;
  double delta_x = 0;

  // matlab
  double new_x;
  if (theta > 0)
    new_x = std::min(theta, lp_.colUpper_[col]);
  else
    new_x = std::max(theta, lp_.colLower_[col]);
  delta_x = new_x - col_value_[col];

  // clp minimizing for delta_x
  // if (theta > 0)
  //   delta_x = std::min(theta, lp_.colUpper_[col] - col_value_[col]);
  // else
  //   delta_x = std::max(theta, lp_.colLower_[col] - col_value_[col]);

  col_value_[col] += delta_x;

  // std::cout << "col " << col << ": " << delta_x << std::endl;

  // Update objective, row_value, residual after each component update.
  objective_ += lp_.colCost_[col] * delta_x;
  for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
    int row = lp_.Aindex_[k];
    residual_[row] -= lp_.Avalue_[k] * delta_x;
    row_value_[row] += lp_.Avalue_[k] * delta_x;
  }

  return delta_x;
}

// // Returns c'x + lambda'x + 1/2mu r'r
// double Quadratic::calculateQuadraticValue(const double mu,
//                                           const std::vector<double> lambda,
//                                           ResidualFunctionType type) {
//   update(type);

//   // c'x
//   double quadratic = getObjective();

//   // lambda'x
//   for (int row = 0; row < lp_.numRow_; row++) {
//     if (type == ResidualFunctionType::kPiecewise) assert(residual_[row] >=
//     0); quadratic += lambda[row] * residual_[row];
//   }

//   // 1/2mu r'r
//   for (int row = 0; row < lp_.numRow_; row++) {
//     quadratic += (residual_[row] * residual_[row]) / mu;
//   }

//   return quadratic;
// }

double Quadratic::findBreakpoints(const int col, const double mu,
                                  const std::vector<double> lambda) {
  // Find breakpoints of residual function and add them to vector.
  std::vector<double> breakpoints;
  for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
    const double entry = lp_.Avalue_[k];
    const int row = lp_.Aindex_[k];
    if (lp_.rowLower_[row] > -HIGHS_CONST_INF) {
      double theta = (lp_.rowLower_[row] - row_value_[row]) / entry;
      breakpoints.push_back(theta);
    }
  }
  for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
    const double entry = lp_.Avalue_[k];
    const int row = lp_.Aindex_[k];
    if (lp_.rowUpper_[row] < HIGHS_CONST_INF) {
      double theta = (lp_.rowUpper_[row] - row_value_[row]) / entry;
      breakpoints.push_back(theta);
    }
  }

  std::sort(breakpoints.begin(), breakpoints.end());
  std::vector<double>::iterator end =
      std::unique(breakpoints.begin(), breakpoints.end());

  double col_save = col_value_[col];

  // // todo change iterators to int
  // std::vector<double>::iterator min_x_update_it_i = breakpoints.begin();
  // std::vector<double>::iterator min_x_update_it_ii = breakpoints.begin();
  // col_value_[col] = col_save + *breakpoints.begin();
  // double current =
  //     calculateQuadraticValue(mu, lambda, ResidualFunctionType::kPiecewise);

  // double min_quadratic_value_i = HIGHS_CONST_INF;
  // double min_quadratic_value_ii = HIGHS_CONST_INF;
  // for (auto it = breakpoints.begin(); it < end; it++) {
  //   col_value_[col] = col_save + *it;
  //   double min =
  //       calculateQuadraticValue(mu, lambda,
  //       ResidualFunctionType::kPiecewise);

  //   if (min < min_quadratic_value_i) {
  //     min_quadratic_value_i = min;
  //     min_quadratic_value_ii = min;
  //     min_x_update_it_i = it;
  //     min_x_update_it_ii = it;

  //   } else if (min < min_quadratic_value_ii) {
  //     min_quadratic_value_ii = min;
  //     min_x_update_it_ii = it;
  //   } else if (fabs(min - min_quadratic_value_i) < 1e08 &&
  //              fabs(min - min_quadratic_value_ii) < 1e08) {
  //     min_x_update_it_ii = it;
  //   }
  // }

  // col_value_[col] = col_save;

  // if (current < min_quadratic_value_i) return 0;

  // // Minimize both quadratics and save min theta's in delta_lhs and
  // delta_rhs. double delta_lhs = 0; double delta_mhs = 0; double delta_rhs =
  // 0;

  // // minimize quadratic ,_,
  // if (min_x_update_it_i != min_x_update_it_ii) {
  //   std::vector<double>::iterator it = min_x_update_it_i;
  //   double left = *min_x_update_it_i;
  //   double right = *min_x_update_it_ii;
  //   assert(left < right);
  //   assert(left > -HIGHS_CONST_INF);
  //   assert(right < HIGHS_CONST_INF);

  //   col_value_[col] = left;
  //   update();

  //   double a = 0.0;
  //   double b = 0.0;

  //   for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
  //     int row = lp_.Aindex_[k];
  //     a += lp_.Avalue_[k] * lp_.Avalue_[k];
  //     double bracket = -residual_[row] - lp_.Avalue_[k] * col_value_[col];
  //     bracket += lambda[row];
  //     b += lp_.Avalue_[k] * bracket;
  //   }

  //   a = (0.5 / mu) * a;
  //   b = (0.5 / mu) * b + 0.5 * lp_.colCost_[col];

  //   double theta = -b / a;  // this is min for x_j

  //   if (theta < left + col_save)
  //     delta_mhs = left + col_save;
  //   else if (theta > right + col_save)
  //     delta_mhs = right + col_save;
  //   else
  //     delta_mhs = theta;
  // }

  // // minimize quadratic `\,
  // std::vector<double>::iterator it = min_x_update_it_i;
  // if (it >= breakpoints.begin()) {
  //   double left = *(it - 1);
  //   double right;
  //   if (it == breakpoints.begin()) {
  //     left = -HIGHS_CONST_INF;
  //     right = *it;
  //     col_value_[col] = right;
  //   } else if (it < end) {
  //     right = *it;
  //     col_value_[col] = right;
  //   } else {
  //     right = HIGHS_CONST_INF;
  //     col_value_[col] = left;
  //   }
  //   update();

  //   double a = 0.0;
  //   double b = 0.0;

  //   for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
  //     int row = lp_.Aindex_[k];
  //     a += lp_.Avalue_[k] * lp_.Avalue_[k];
  //     double bracket = -residual_[row] - lp_.Avalue_[k] * col_value_[col];
  //     bracket += lambda[row];
  //     b += lp_.Avalue_[k] * bracket;
  //   }

  //   a = (0.5 / mu) * a;
  //   b = (0.5 / mu) * b + 0.5 * lp_.colCost_[col];

  //   double theta = -b / a;  // this is min for x_j
  //   theta = theta - col_value_[col];

  //   if (theta < left + col_save)
  //     delta_lhs = left;
  //   else if (theta > right + col_save)
  //     delta_lhs = right;
  //   else
  //     delta_lhs = theta;
  // }

  // // minimize quadratic ,/`
  // it = min_x_update_it_ii;
  // if (it <= end - 1) {
  //   double left = *it;
  //   double right = *(it + 1);
  //   if (it == breakpoints.begin()) {
  //   } else if (it == end - 1) {
  //     right = HIGHS_CONST_INF;
  //   }

  //   col_value_[col] = left;
  //   update();

  //   double a = 0.0;
  //   double b = 0.0;

  //   for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
  //     int row = lp_.Aindex_[k];
  //     a += lp_.Avalue_[k] * lp_.Avalue_[k];
  //     double bracket = -residual_[row] - lp_.Avalue_[k] * col_value_[col];
  //     bracket += lambda[row];
  //     b += lp_.Avalue_[k] * bracket;
  //   }

  //   a = (0.5 / mu) * a;
  //   b = (0.5 / mu) * b + 0.5 * lp_.colCost_[col];

  //   double theta = -b / a;  // this is min for x_j
  //   theta = theta - col_value_[col];
  //   if (theta < left + col_save)
  //     delta_rhs = left + col_save;
  //   else if (theta > right + col_save)
  //     delta_rhs = right + col_save;
  //   else
  //     delta_rhs = theta;
  // }

  // col_value_[col] = col_save + delta_lhs;
  // double left_min_value =
  //     calculateQuadraticValue(mu, lambda, ResidualFunctionType::kPiecewise);
  // col_value_[col] = col_save + delta_mhs;
  // double mid_min_value =
  //     calculateQuadraticValue(mu, lambda, ResidualFunctionType::kPiecewise);
  // col_value_[col] = col_save + delta_rhs;
  // double right_min_value =
  //     calculateQuadraticValue(mu, lambda, ResidualFunctionType::kPiecewise);
  // col_value_[col] = col_save;

  // double min_value;
  // if (min_x_update_it_i != min_x_update_it_ii) {
  //   if (left_min_value < right_min_value && left_min_value < mid_min_value)
  //     min_value = delta_lhs;
  //   else if (right_min_value < left_min_value &&
  //            right_min_value < mid_min_value)
  //     min_value = delta_rhs;
  //   else
  //     min_value = delta_mhs;
  // } else if (left_min_value < right_min_value)
  //   min_value = delta_lhs;
  // else
  //   min_value = delta_rhs;

  // if ((col_value_[col] + min_value) < lp_.colLower_[col])
  //   min_value = lp_.colLower_[col] - col_value_[col];
  // if ((col_value_[col] + min_value) > lp_.colUpper_[col])
  //   min_value = lp_.colUpper_[col] - col_value_[col];

  // return min_value - col_save;
  return 0;
}

double Quadratic::minimize_component_quadratic_piecewise(
    const int col, const double mu, const std::vector<double>& lambda) {
  // Calculate step theta using true residual. The function minimized for each
  // component has breakpoints. They are found and sorted and the minimum
  // Quadratic function value of them is taken as an approximation of the true
  // minimum.

  double theta = findBreakpoints(col, mu, lambda);

  // matlab
  double new_x;
  if (theta > 0)
    new_x = std::min(theta, lp_.colUpper_[col] - col_value_[col]);
  else
    new_x = std::max(theta, col_value_[col] - lp_.colLower_[col]);
  double delta_x = new_x - col_value_[col];

  col_value_[col] += delta_x;

  // std::cout << "col " << col << ": " << delta_x << std::endl;

  // Update objective, row_value, residual after each component update.
  objective_ += lp_.colCost_[col] * delta_x;
  for (int k = lp_.Astart_[col]; k < lp_.Astart_[col + 1]; k++) {
    int row = lp_.Aindex_[k];
    // todo: this is slow
    update(ResidualFunctionType::kPiecewise);
  }
  return delta_x;
}

void Quadratic::minimize_by_component(
    const double mu, const std::vector<double>& lambda,
    const ResidualFunctionType quadratic_type) {
  HighsPrintMessageLevel ML_DESC = ML_DETAILED;
  int iterations = 100;
  bool minor_iteration_details = false;

  HighsPrintMessage(ML_DESC, "Values at start: %3.2g, %3.4g, \n", objective_,
                    residual_norm_2_);

  for (int iteration = 0; iteration < iterations; iteration++) {
    for (int col = 0; col < lp_.numCol_; col++) {
      // determine whether to minimize for col.
      // if empty skip.
      if (lp_.Astart_[col] == lp_.Astart_[col + 1]) continue;

      double delta_x = 0;
      if (quadratic_type == ResidualFunctionType::kLinearised)
        delta_x = minimize_component_quadratic_linearisation(col, mu, lambda);
      else if (quadratic_type == ResidualFunctionType::kPiecewise)
        delta_x = minimize_component_quadratic_piecewise(col, mu, lambda);

      if (minor_iteration_details) {
        double quadratic_objective = getQuadraticObjective(
            lp_.colCost_, col_value_, residual_, mu, lambda);

        printMinorIterationDetails(iteration, col, col_value_[col] - delta_x,
                                   delta_x, objective_, residual_,
                                   quadratic_objective);
      }
    }

    // todo: update(type)? when you get back to breakpoints.
    update();

    HighsPrintMessage(ML_DESC,
                      "Values at approximate iteration %d: %3.2g, %3.4g, \n",
                      iteration, objective_, residual_norm_2_);

    // todo: check for early exit
  }
  update();
}

HighsStatus runFeasibility(const HighsLp& lp, HighsSolution& solution,
                           const MinimizationType type,
                           const double initial_weight, HighsTimer& timer) {
  if (!isEqualityProblem(lp)) return HighsStatus::Error;
  if (type == MinimizationType::kExactAdmm) return HighsStatus::Error;
  if (type == MinimizationType::kComponentWiseAdmm)
    return HighsStatus::Error;

  double start_time = timer.getWallTime();

  solution.col_value.clear();
  solution.col_value.resize(lp.numCol_);

  // todo: add test
  // for an equality problem kLinearized and kPiecewise should give the
  // same result.

  ResidualFunctionType residual_type =
      (type != MinimizationType::kComponentWiseBreakpoints)
          ? ResidualFunctionType::kLinearised
          : ResidualFunctionType::kPiecewise;

  if (residual_type != ResidualFunctionType::kPiecewise &&
      !isEqualityProblem(lp)) {
    HighsPrintMessage(
        ML_ALWAYS,
        "Error: FindFeasibility does not support smooth minimization of "
        "inequality problems. Run transformIntoEqualityProblem(..) before "
        "runFeasibility(..)\n");
    return HighsStatus::Error;
  }

  if (lp.sense_ != OBJSENSE_MINIMIZE) {
    HighsPrintMessage(
        ML_ALWAYS,
        "Error: FindFeasibility does not support maximization problems.\n");
    return HighsStatus::Error;
  }

  // Initialize x_0 ≥ 0, μ_1, λ_1 = 0.
  double mu = initial_weight;
  std::vector<double> lambda;

  HighsStatus status = initialize(lp, solution, mu, lambda);
  if (status != HighsStatus::OK) {
    // todo: handle errors.
  }

  Quadratic quadratic(lp, solution.col_value, residual_type);
  // Quadratic quadratic(lp, solution.col_value);

  if (type == MinimizationType::kComponentWise)
    HighsPrintMessage(ML_ALWAYS,
                      "Minimizing quadratic subproblem component-wise...\n");
  else if (type == MinimizationType::kExact)
    HighsPrintMessage(ML_ALWAYS,
                      "Minimizing quadratic subproblem exactly...\n");

  // Report values at start.
  std::stringstream ss;
  double residual_norm_2 = quadratic.getResidualNorm2();
  ss << "Iteration " << std::setw(3) << 0 << ": objective " << std::setw(3)
     << std::fixed << std::setprecision(2) << quadratic.getObjective()
     << " residual " << std::setw(5) << std::scientific
     << quadratic.getResidualNorm2() << std::endl;
  HighsPrintMessage(ML_ALWAYS, ss.str().c_str());

  residual_norm_2 = quadratic.getResidualNorm2();
  if (residual_norm_2 < kExitTolerance) {
    HighsPrintMessage(ML_ALWAYS,
                      "Solution feasible within exit tolerance: %g.\n",
                      kExitTolerance);
    return HighsStatus::OK;
  }

  // Minimize approximately for K iterations.
  int K = 30;
  int iteration = 0;
  for (iteration = 1; iteration < K + 1; iteration++) {
    // Minimize quadratic function.
    switch (type) {
      case MinimizationType::kComponentWise:
        quadratic.minimize_by_component(mu, lambda,
                                        ResidualFunctionType::kLinearised);
        break;
      case MinimizationType::kComponentWiseBreakpoints:
        quadratic.minimize_by_component(mu, lambda,
                                        ResidualFunctionType::kPiecewise);
        break;
      case MinimizationType::kExact:
        quadratic.minimize_exact_with_lambda(mu, lambda);
        break;
      case MinimizationType::kExactPenalty:
        quadratic.minimize_exact_penalty(mu);
        break;
    }

    // Report outcome.
    residual_norm_2 = quadratic.getResidualNorm2();
    ss.str(std::string());
    bool details = false;
    if (!details) {
      ss << "Iteration " << std::setw(3) << iteration << ": objective "
         << std::setw(3) << std::fixed << std::setprecision(2)
         << quadratic.getObjective() << " residual " << std::setw(5)
         << std::scientific << residual_norm_2 << std::endl;
    } else {
      // todo: after merge with ff-breakpoints
      // or before and replace code there
      // double quad_value = calculateQuadraticValue(A,L,U,x,residual_type);
      ss << "Iter " << std::setw(3) << iteration << ", c'x "
         << std::setprecision(5) << quadratic.getObjective() << ", res "
         << residual_norm_2 << std::endl;
    }
    HighsPrintMessage(ML_ALWAYS, ss.str().c_str());

    // Exit if feasible.
    if (residual_norm_2 < kExitTolerance) {
      HighsPrintMessage(ML_ALWAYS,
                        "Solution feasible within exit tolerance: %g.\n",
                        kExitTolerance);
      break;
    }

    // Update mu every third iteration, otherwise update lambda.
    if (iteration % 3 == 2) {
      mu = 0.1 * mu;
    } else {
      lambda = quadratic.getResidual();
      for (int row = 0; row < lp.numRow_; row++) lambda[row] = mu * lambda[row];
    }
  }

  quadratic.getSolution(solution);
  HighsPrintMessage(ML_ALWAYS,
                    "\nSolution set at the end of feasibility search.\n");

  // Using ss again instead of ss_str messes up HighsIO.
  assert(calculateObjective(lp, solution) == quadratic.getObjective());
  std::vector<double> r;
  status = calculateResidual(lp, solution, r);
  bool is_ok = (status == HighsStatus::OK);
  assert(is_ok);
  assert(std::fabs(residual_norm_2 - getNorm2(r)) < 1e08);

  double quadratic_objective =
      getQuadraticObjective(lp.colCost_, solution.col_value, r, mu, lambda);

  double end_time = timer.getWallTime();
  double time = end_time - start_time;

  std::stringstream ss_str;
  ss_str << "Model, " << lp.model_name_ << ", iter, " << iteration << ", mu, "
         << mu << ", c'x, " << std::scientific << quadratic.getObjective()
         << ", quadratic_objective, " << std::setw(3) << quadratic_objective
         << ", residual, " << std::setw(5) << residual_norm_2 << ", time, "
         << time << std::endl;
  HighsPrintMessage(ML_ALWAYS, ss_str.str().c_str());

  return HighsStatus::OK;
}
