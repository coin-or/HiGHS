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

#ifdef CUPDLP_GPU
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// #include "Highs.h"
#include "linalg.hpp"
#include "logger.hpp"
#include "pdlp/HiPdlpTimer.h"
#include "restart.hpp"
#include "scaling.hpp"
#include "solver_results.hpp"

// Forward declaration for a struct defined in the .cc file
struct StepSizeConfig;

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

#ifdef CUPDLP_GPU
  // --- GPU methods ---
  void setupGpu();
  void cleanupGpu();
  void linalgGpuAx(const double* d_x_in, double* d_ax_out);
  void linalgGpuATy(const double* d_y_in, double* d_aty_out);

// --- Helpers for error checking ---
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

#define CUSPARSE_CHECK(call)                                               \
  do {                                                                     \
    cusparseStatus_t status = (call);                                      \
    if (status != CUSPARSE_STATUS_SUCCESS) {                               \
      fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cusparseGetErrorString(status));                             \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(call)                                               \
  do {                                                                   \
    cublasStatus_t status = call;                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      fprintf(stderr, "cuBLAS Error at %s:%d: %d\n", __FILE__, __LINE__, \
              status);                                                   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)
  // --- GPU Members ---
  cusparseHandle_t cusparse_handle_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;

  // Matrix A in CSR format (for Ax)
  cusparseSpMatDescr_t mat_a_csr_ = nullptr;
  int* d_a_row_ptr_ = nullptr;
  int* d_a_col_ind_ = nullptr;
  double* d_a_val_ = nullptr;

  // Matrix A^T in CSR format (for Aty)
  cusparseSpMatDescr_t mat_a_T_csr_ = nullptr;
  int* d_at_row_ptr_ = nullptr;
  int* d_at_col_ind_ = nullptr;
  double* d_at_val_ = nullptr;

  // GPU vectors
  int a_num_rows_ = 0;
  int a_num_cols_ = 0;
  int a_nnz_ = 0;
  double* d_col_cost_ = nullptr;
  double* d_col_lower_ = nullptr;
  double* d_col_upper_ = nullptr;
  double* d_row_lower_ = nullptr;
  bool* d_is_equality_row_ = nullptr;
  double* d_x_current_ = nullptr;
  double* d_y_current_ = nullptr;
  double* d_x_avg_ = nullptr;
  double* d_y_avg_ = nullptr;
  double* d_x_next_ = nullptr;
  double* d_y_next_ = nullptr;
  double* d_x_at_last_restart_ = nullptr;
  double* d_y_at_last_restart_ = nullptr;
  double* d_x_temp_diff_norm_result_ = nullptr;
  double* d_y_temp_diff_norm_result_ =
      nullptr;                       // Temporary buffer for reduction result
  double* d_ax_current_ = nullptr;   // Replaces host-side Ax_cache_
  double* d_aty_current_ = nullptr;  // Replaces host-side ATy_cache_
  double* d_ax_next_ = nullptr;
  double* d_aty_next_ = nullptr;
  double* d_ax_avg_ = nullptr;
  double* d_aty_avg_ = nullptr;
  double* d_x_sum_ = nullptr;
  double* d_y_sum_ = nullptr;

  // States
  double sum_weights_gpu_ = 0.0;
  double* d_convergence_results_ = nullptr;  // size 4
  double* d_dSlackPos_ = nullptr;
  double* d_dSlackNeg_ = nullptr;
  double* d_dSlackPosAvg_ = nullptr;
  double* d_dSlackNegAvg_ = nullptr;
  double* d_col_scale_ = nullptr;
  double* d_row_scale_ = nullptr;
  bool checkConvergenceGpu(const int iter, const double* d_x, const double* d_y,
                           const double* d_ax, const double* d_aty,
                           double epsilon, SolverResults& results,
                           const char* type);

  // Temporary buffer for SpMV
  void* d_spmv_buffer_ax_ = nullptr;
  size_t spmv_buffer_size_ax_ = 0;
  void* d_spmv_buffer_aty_ = nullptr;
  size_t spmv_buffer_size_aty_ = 0;
  double* d_buffer_;  // for cublas
  double* d_buffer2_;

  void launchKernelUpdateX(double primal_step);
  void launchKernelUpdateY(double dual_step);
  void launchKernelUpdateAverages(double weight);
  void launchKernelScaleVector(double* d_out, const double* d_in, double scale,
                               int n);
  void computeStepSizeRatioGpu(PrimalDualParams& working_params);
  void updateAverageIteratesGpu(int inner_iter);
  void computeAverageIterateGpu();
  double computeMovementGpu(const double* d_x_new, const double* d_x_old,
                            const double* d_y_new, const double* d_y_old);

  double computeNonlinearityGpu(const double* d_x_new, const double* d_x_old,
                                const double* d_aty_new,
                                const double* d_aty_old);
  double computeDiffNormCuBLAS(const double* d_a, const double* d_b, int n);

  cusparseDnVecDescr_t vec_x_desc_ = nullptr;
  cusparseDnVecDescr_t vec_y_desc_ = nullptr;
  cusparseDnVecDescr_t vec_ax_desc_ = nullptr;
  cusparseDnVecDescr_t vec_aty_desc_ = nullptr;
#endif
};

#endif
