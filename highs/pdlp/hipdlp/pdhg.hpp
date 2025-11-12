/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/hipdlp/pdhg.hpp
 * @brief
 */
#ifndef PDHG_HPP
#define PDHG_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "Highs.h"
#include "linalg.hpp"
#include "logger.hpp"
#include "pdlp/HiPdlpTimer.h"
#include "restart.hpp"
#include "scaling.hpp"
#include "solver_results.hpp"

// Forward declaration for a struct defined in the .cc file
struct cusparseContext;
struct cublasContext;
struct cusparseSpMatDescr;
struct cusparseDnVecDescr;
struct StepSizeConfig;

/*
class PDLPSolver {
 public:
  // --- Public API ---
  void setup(const HighsOptions& options, HighsTimer& timer);
  void passLp(const HighsLp* lp) { original_lp_ = lp; }
  void preprocessLp();
  void scaleProblem();
  void solve(std::vector<double>& x, std::vector<double>& y);
  void unscaleSolution(std::vector<double>& x, std::vector<double>& y);
  PostSolveRetcode postprocess(HighsSolution& solution);
  void logSummary();

  // --- Getters ---
  TerminationStatus getTerminationCode() const { return results_.term_code; }
  int getIterationCount() const { return final_iter_count_; }
  int getnCol() const { return lp_.num_col_; }
  int getnRow() const { return lp_.num_row_; }

  // --- Debugging ---
  FILE* debug_pdlp_log_file_ = nullptr;
  DebugPdlpData debug_pdlp_data_;

  void reportHipdlpTimer();
  void closeDebugLog();

 private:
  // --- Core Algorithm Logic ---
  void solveReturn(const TerminationStatus term_code);
  void initialize();
  void printConstraintInfo();
  bool checkConvergence(const int iter, const std::vector<double>& x,
                        const std::vector<double>& y,
                        const std::vector<double>& ax_vector,
                        const std::vector<double>& aty_vector, double epsilon,
                        SolverResults& results, const char* type,
                        // Add slack vectors as non-const references
                        std::vector<double>& dSlackPos,
                        std::vector<double>& dSlackNeg);
  void updateAverageIterates(const std::vector<double>& x,
                             const std::vector<double>& y,
                             const PrimalDualParams& params, int inner_iter);
  void computeAverageIterate(std::vector<double>& ax_avg,
                             std::vector<double>& aty_avg);
  double PowerMethod();

  // --- Step update Methods (previously in Step) ---
  void initializeStepSizes();
  std::vector<double> updateX(const std::vector<double>& x,
                              const std::vector<double>& aty,
                              double primal_step);
  std::vector<double> updateY(const std::vector<double>& y,
                              const std::vector<double>& ax,
                              const std::vector<double>& ax_next,
                              double dual_step);
  void updateIteratesFixed();
  void updateIteratesAdaptive();
  bool updateIteratesMalitskyPock(bool first_malitsky_iteration);

  // --- Step Size Helper Methods (previously in PdlpStep) ---
  bool CheckNumericalStability(const std::vector<double>& delta_x,
                               const std::vector<double>& delta_y);
  double computeMovement(const std::vector<double>& delta_primal,
                         const std::vector<double>& delta_dual);
  double computeNonlinearity(const std::vector<double>& delta_primal,
                             const std::vector<double>& delta_aty);

  // --- Feasibility, Duality, and KKT Checks ---
  std::vector<double> computeLambda(const std::vector<double>& y,
                                    const std::vector<double>& ATy_vector);
  double computeDualObjective(const std::vector<double>& y,
                              const std::vector<double>& dSlackPos,
                              const std::vector<double>& dSlackNeg);
  double computePrimalFeasibility(const std::vector<double>& Ax_vector);
  void computeDualSlacks(const std::vector<double>& dualResidual,
                         std::vector<double>& dSlackPos,
                         std::vector<double>& dSlackNeg);
  double computeDualFeasibility(const std::vector<double>& ATy_vector,
                                std::vector<double>& dSlackPos,
                                std::vector<double>& dSlackNeg);
  std::tuple<double, double, double, double, double> computeDualityGap(
      const std::vector<double>& x, const std::vector<double>& y,
      const std::vector<double>& lambda);
  void computeStepSizeRatio(PrimalDualParams& working_params);
  void hipdlpTimerStart(const HighsInt hipdlp_clock);
  void hipdlpTimerStop(const HighsInt hipdlp_clock);

  // --- Problem Data and Parameters ---
  HighsLp lp_;
  const HighsLp* original_lp_;
  HighsLp unscaled_processed_lp_;
  PrimalDualParams params_;
  StepSizeConfig stepsize_;
  Logger logger_;
  HighsLogOptions log_options_;
  SolverResults results_;
  int original_num_col_;
  int num_eq_rows_;
  std::vector<bool> is_equality_row_;
  std::vector<int> constraint_new_idx_;
  std::vector<ConstraintType> constraint_types_;
  int sense_origin_ = 1;

  // --- Solver State ---
  int final_iter_count_ = 0;
  std::vector<double> x_current_, y_current_;
  std::vector<double> x_next_, y_next_;
  std::vector<double> x_avg_, y_avg_;
  std::vector<double> x_sum_, y_sum_;
  double sum_weights_ = 0.0;
  double current_eta_ = 0.0;
  double ratio_last_two_step_sizes_ = 1.0;
  int num_rejected_steps_ = 0;
  std::vector<double> dSlackPos_;
  std::vector<double> dSlackNeg_;
  std::vector<double> dSlackPosAvg_;
  std::vector<double> dSlackNegAvg_;
  Timer total_timer;

  HipdlpTimer hipdlp_timer_;
  HighsTimerClock hipdlp_clocks_;

  // --- Scaling ---
  Scaling scaling_;
  double unscaled_rhs_norm_ = 0.0;
  double unscaled_c_norm_ = 0.0;

  // --- Restarting ---
  RestartScheme restart_scheme_;
  std::vector<double> x_at_last_restart_;
  std::vector<double> y_at_last_restart_;

  // --- Caching for Matrix-Vector Products ---
  std::vector<double> Ax_cache_;
  std::vector<double> ATy_cache_;
  std::vector<double> Ax_next_, ATy_next_;
  std::vector<double> K_times_x_diff_;
};
*/

class PDLPSolver {
 public:
  // --- Public API ---
  void setup(const HighsOptions& options, HighsTimer& timer);
  void passLp(const HighsLp* lp) { original_lp_ = lp; }
  void preprocessLp();
  void scaleProblem();
  void solve(std::vector<double>& x, std::vector<double>& y);
  void unscaleSolution(std::vector<double>& x, std::vector<double>& y);
  PostSolveRetcode postprocess(HighsSolution& solution);
  void logSummary();

  // --- Getters ---
  TerminationStatus getTerminationCode() const { return results_.term_code; }
  int getIterationCount() const { return final_iter_count_; }
  int getnCol() const { return lp_.num_col_; }
  int getnRow() const { return lp_.num_row_; }

  // --- Debugging ---
  FILE* debug_pdlp_log_file_ = nullptr;
  DebugPdlpData debug_pdlp_data_;

  void reportHipdlpTimer();
  void closeDebugLog();

 private:
  // --- Core Algorithm Logic ---
  void solveReturn(const TerminationStatus term_code);
  void initialize();
  void printConstraintInfo();
  bool checkConvergence(const int iter, const PdlpVector& x,
                        const PdlpVector& y,
                        const PdlpVector& ax_vector,
                        const PdlpVector& aty_vector, double epsilon,
                        SolverResults& results, const char* type,
                        PdlpVector& dSlackPos, PdlpVector& dSlackNeg);
  void updateAverageIterates(const PdlpVector& x, const PdlpVector& y,
                             const PrimalDualParams& params, int inner_iter);
  void computeAverageIterate(PdlpVector& ax_avg, PdlpVector& aty_avg);
  double PowerMethod();

  // --- Step update Methods (previously in Step) ---
  void initializeStepSizes();
  void updateIteratesFixed();
  void updateIteratesAdaptive();
  bool updateIteratesMalitskyPock(bool first_malitsky_iteration);

  // --- Step Size Helper Methods (previously in PdlpStep) ---
  bool CheckNumericalStability(const PdlpVector& delta_x,
                               const PdlpVector& delta_y);
  double computeMovement(const PdlpVector& delta_primal,
                         const PdlpVector& delta_dual);
  double computeNonlinearity(const PdlpVector& delta_primal,
                             const PdlpVector& delta_aty);

  // --- Feasibility, Duality, and KKT Checks ---
  void computeStepSizeRatio(PrimalDualParams& working_params);
  void hipdlpTimerStart(const HighsInt hipdlp_clock);
  void hipdlpTimerStop(const HighsInt hipdlp_clock);

  // --- Problem Data and Parameters ---
  HighsLp lp_;
  const HighsLp* original_lp_;
  HighsLp unscaled_processed_lp_;
  PrimalDualParams params_;
  StepSizeConfig stepsize_;
  Logger logger_;
  HighsLogOptions log_options_;
  SolverResults results_;
  int original_num_col_;
  int num_eq_rows_;
  std::vector<bool> is_equality_row_;
  std::vector<int> constraint_new_idx_;
  std::vector<ConstraintType> constraint_types_;
  int sense_origin_ = 1;

  // --- Backend and Device Selection ---
  Device device_ = Device::CPU;
  std::unique_ptr<LinearAlgebraBackend> backend_;

  // --- Solver State ---
  int final_iter_count_ = 0;
  std::unique_ptr<PdlpVector> x_current_, y_current_;
  std::unique_ptr<PdlpVector> x_next_, y_next_;
  std::unique_ptr<PdlpVector> x_avg_, y_avg_;
  std::unique_ptr<PdlpVector> x_sum_, y_sum_;
  std::unique_ptr<PdlpVector> x_diff_temp_;
  std::unique_ptr<PdlpVector> y_diff_temp_;
  double sum_weights_ = 0.0;
  double current_eta_ = 0.0;
  double ratio_last_two_step_sizes_ = 1.0;
  int num_rejected_steps_ = 0;
  Timer total_timer;
  
  // --- Device-side slack vectors ---
  std::unique_ptr<PdlpVector> dSlackPos_;
  std::unique_ptr<PdlpVector> dSlackNeg_;
  std::unique_ptr<PdlpVector> dSlackPosAvg_;
  std::unique_ptr<PdlpVector> dSlackNegAvg_;

  // --- Host-side buffers for postprocessing ---
  std::vector<double> h_dSlackPos_;
  std::vector<double> h_dSlackNeg_;

  HipdlpTimer hipdlp_timer_;
  HighsTimerClock hipdlp_clocks_;

  // --- Scaling ---
  Scaling scaling_;
  double unscaled_rhs_norm_ = 0.0;
  double unscaled_c_norm_ = 0.0;

  // --- Restarting ---
  RestartScheme restart_scheme_;
  std::unique_ptr<PdlpVector> x_at_last_restart_;
  std::unique_ptr<PdlpVector> y_at_last_restart_;
  // Host vectors
  std::vector<double> h_x_current_, h_x_at_last_restart_;
  std::vector<double> h_y_current_, h_y_at_last_restart_;

  // --- Caching for Matrix-Vector Products ---
  std::unique_ptr<PdlpVector> Ax_cache_;
  std::unique_ptr<PdlpVector> ATy_cache_;
  std::unique_ptr<PdlpVector> Ax_next_, ATy_next_;
  std::unique_ptr<PdlpVector> Ax_avg_, ATy_avg_;

  // --- Problem Data (Device-side) ---
  std::unique_ptr<PdlpSparseMatrix> matrix_;
  std::unique_ptr<PdlpVector> col_cost_vec_;
  std::unique_ptr<PdlpVector> col_lower_vec_;
  std::unique_ptr<PdlpVector> col_upper_vec_;
  std::unique_ptr<PdlpVector> rhs_vec_;
  std::unique_ptr<PdlpVector> is_equality_row_vec_;

  // --- Temporary Vectors for Convergence Checks ---
  std::unique_ptr<PdlpVector> primal_residual_;
  std::unique_ptr<PdlpVector> dual_residual_;
  std::unique_ptr<PdlpVector> dual_residual_avg_;
};

#endif
