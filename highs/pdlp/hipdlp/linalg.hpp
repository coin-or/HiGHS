/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/hipdlp/linalg.hpp
 * @brief
 */
#ifndef PDLP_HIPDLP_LINALG_HPP
#define PDLP_HIPDLP_LINALG_HPP

#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include "Highs.h"

enum class Device { CPU, GPU };

// --- Abstract Data Handles ---
// The solver will only interact with these abstract types.

/**
 * @brief Abstract vector handle.
 *
 * Provides an interface for a device-specific vector (e.g., std::vector
 * on CPU or a device pointer on GPU).
 */
class PdlpVector {
public:
    virtual ~PdlpVector() = default;
    virtual void copyFromHost(const std::vector<double>& host_data) = 0;
    virtual void copyToHost(std::vector<double>& host_data) const = 0;
    virtual void copyFrom(const PdlpVector& other) = 0; // Device to Device or Host to Host
    virtual void fill(double value) = 0;
    virtual size_t size() const = 0;
};

/**
 * @brief Abstract sparse matrix handle.
 *
 * Provides an interface for a device-specific sparse matrix
 * (e.g., HighsSparseMatrix on CPU or cuSPARSE descriptors on GPU).
 */
class PdlpSparseMatrix {
public:
  virtual ~PdlpSparseMatrix() = default;

  virtual size_t num_rows() const = 0;
  virtual size_t num_cols() const = 0;
};

// --- Abstract Backend Interface ---

/**
 * @brief Abstract interface for all linear algebra operations.
 *
 * This class defines the complete set of mathematical operations
 * required by the PDLPSolver, allowing for runtime switching
 * between CPU and GPU implementations.
 */
class LinearAlgebraBackend {
public:
    virtual ~LinearAlgebraBackend() = default;

    // --- Factory Methods ---
    // The backend is responsible for creating its own compatible data types.
    virtual std::unique_ptr<PdlpVector> createVector(size_t size) const = 0;
    virtual std::unique_ptr<PdlpSparseMatrix> createSparseMatrix(
        const HighsLp& lp) const = 0;
    // virtual std::unique_ptr<PdlpSparseMatrix> createBlockSparseMatrix(...) const = 0;
    
    // --- Core Linear Algebra ---
    virtual void Ax(PdlpVector& results, const PdlpSparseMatrix& A, const PdlpVector& x) const = 0;
    virtual void ATy(PdlpVector& results, const PdlpSparseMatrix& A, const PdlpVector& y) const = 0;

    // --- BLAS-like Operations ---
    /** @brief result = a^T * b */
    virtual double dot(const PdlpVector& a, const PdlpVector& b) const = 0;

    /** @brief result = ||a||_2 */
    virtual double nrm2(const PdlpVector& a) const = 0;

    /** @brief result = ||a||_p */
    virtual double vector_norm(const PdlpVector& vec, double p = 2.0) const = 0;
    
    /** @brief a = factor * a */
    virtual void scale(PdlpVector& a, double factor) const = 0;

    /** @brief result = a - b */
    virtual void sub(PdlpVector& result, const PdlpVector& a,
                    const PdlpVector& b) const = 0;
    
    // --- PDHG-Specific Kernels ---                
    /**
     * @brief Performs the primal update step (gradient and projection).
     * x_new = project_box(x_old - primal_step * (c - aty), l, u)
     */
    virtual void updateX(PdlpVector& x_new, const PdlpVector& x_old,
                        const PdlpVector& aty, const PdlpVector& c,
                        const PdlpVector& l, const PdlpVector& u,
                        double primal_step) const = 0;

    /**
     * @brief Performs the dual update step (gradient and projection).
     * extr_ax = 2 * ax_next - ax
     * dual_update = y_old + dual_step * (rhs - extr_ax)
     * y_new = project_non_negative(dual_update, is_equality_row_vec)
     */
    virtual void updateY(PdlpVector& y_new, const PdlpVector& y_old,
                        const PdlpVector& ax, const PdlpVector& ax_next,
                        const PdlpVector& rhs,
                        const PdlpVector& is_equality_row_vec,
                        double dual_step) const = 0;

    // --- Averaging Kernels ---

    /** @brief sum_vec += weight * vec */
    virtual void accumulate_weighted_sum(PdlpVector& sum_vec,
                                        const PdlpVector& vec,
                                        double weight) const = 0;

    /** @brief avg_vec = sum_vec / total_weight */
    virtual void compute_average(PdlpVector& avg_vec, const PdlpVector& sum_vec,
                                double total_weight) const = 0;

    // --- Convergence Kernels ---
    
    /**
     * @brief Computes dual slacks: dSlackPos = max(0, c - A'y) (for lower)
     * and dSlackNeg = max(0, A'y - c) (for upper).
     * @param dualResidual A temporary vector (pre-allocated)
     */
    virtual void computeDualSlacks(PdlpVector& dSlackPos, PdlpVector& dSlackNeg,
                                    const PdlpVector& c, const PdlpVector& aty,
                                    const PdlpVector& col_lower,
                                    const PdlpVector& col_upper,
                                    PdlpVector& dualResidual) const = 0;
    
    /**
     * @brief Computes || (Ax - b)_[inequality_rows]^+ ||_2
     */
    virtual double computePrimalFeasibility(
        const PdlpVector& ax, const PdlpVector& rhs,
        const PdlpVector& is_equality_row_vec,
        const std::vector<double>& row_scale,
        PdlpVector& primalResidual) const = 0;

    /**
     * @brief Computes || c - A'y - dSlackPos + dSlackNeg ||_2
     */
    virtual double computeDualFeasibility(
        const PdlpVector& c, const PdlpVector& aty,
        const PdlpVector& dSlackPos, const PdlpVector& dSlackNeg,
        const std::vector<double>& col_scale,
        PdlpVector& dualResidual) const = 0;

    /**
     * @brief Computes primal (c'x) and dual (b'y + l's+ - u's-) objectives.
     * @return_param primal_obj Host variable to store primal objective.
     * @return_param dual_obj Host variable to store dual objective.
     */
    virtual void computeObjectives(
        const PdlpVector& c, const PdlpVector& x, const PdlpVector& rhs,
        const PdlpVector& y, const PdlpVector& col_lower,
        const PdlpVector& col_upper, const PdlpVector& dSlackPos,
        const PdlpVector& dSlackNeg, double offset, double& primal_obj,
        double& dual_obj) const = 0;

    // --- Synchronization ---

    /**
     * @brief Blocks until all pending operations on the device are complete.
     * (No-op for CPU, cudaDeviceSynchronize for GPU).
     */
    virtual void synchronize() const = 0;
};

/**
 * @brief Factory function to create the appropriate backend at runtime.
 */
std::unique_ptr<LinearAlgebraBackend> createBackend(Device device);

namespace linalg {
double project_box(double x, double l, double u);
double project_non_negative(double y);
void project_bounds(const HighsLp& lp, std::vector<double>& x);

// Function to compute A*x for a given HighsLp and vector x
void Ax(const HighsLp& lp, const std::vector<double>& x,
        std::vector<double>& result);

// Function to compute A^T*y for a given HighsLp and vector y
void ATy(const HighsLp& lp, const std::vector<double>& y,
         std::vector<double>& result);

double nrm2(const std::vector<double>& vec);
void scale(std::vector<double>& vec, double factor);

void normalize(std::vector<double>& vec);

double dot(const std::vector<double>& a, const std::vector<double>& b);

double diffTwoNorm(const std::vector<double>& v1,
                   const std::vector<double>& v2);

// General norm functions
double vector_norm(const std::vector<double>& vec, double p = 2.0);
double vector_norm(const double* values, size_t size, double p = 2.0);
double vector_norm_squared(const std::vector<double>& vec);

// LP-specific norm calculations
double compute_cost_norm(const HighsLp& lp, double p = 2.0);
double compute_rhs_norm(const HighsLp& lp, double p = 2.0);

// Matrix column/row norm calculations
std::vector<double> compute_column_norms(
    const HighsLp& lp, double p = std::numeric_limits<double>::infinity());
std::vector<double> compute_row_norms(
    const HighsLp& lp, double p = std::numeric_limits<double>::infinity());

std::vector<double> vector_subtrac(const std::vector<double>& a,
                                   const std::vector<double>& b);

}  // namespace linalg

#endif  // PDLP_HIPDLP_LINALG_HPP
