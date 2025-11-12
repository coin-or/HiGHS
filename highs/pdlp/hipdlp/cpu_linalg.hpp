/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/hipdlp/cpu_linalg.hpp
 * @brief CPU implementation of the linear algebra interface.
 */
#ifndef PDLP_HIPDLP_CPU_LINALG_HPP
#define PDLP_HIPDLP_CPU_LINALG_HPP

#include "linalg.hpp"
#include "Highs.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace highs {
namespace pdlp {

// --- Helper functions (from original linalg.cc) ---
static double project_box(double x, double l, double u) {
  return std::max(l, std::min(x, u));
}

static double project_non_negative(double x) { return std::max(0.0, x); }

// --- CPU Vector Implementation ---
class CpuVector : public PdlpVector {
    std::vector<double> data_;
public:
    explicit CpuVector(size_t size) : data_(size, 0.0) {}

    void copyFromHost(const std::vector<double>& host_data) override {
        if (host_data.size() != data_.size()) {
            throw std::invalid_argument("Size mismatch in copyFromHost");
        }
        data_ = host_data;
    }

    void copyToHost(std::vector<double>& host_data) const override {
        if (host_data.size() != data_.size()) {
            host_data.resize(data_.size());
        }
        host_data = data_;
    }
    void copyFrom(const PdlpVector& other) override {
    const auto* other_cpu = dynamic_cast<const CpuVector*>(&other);
    if (!other_cpu) {
        throw std::runtime_error("Cannot copy from non-CPU vector to CPU vector");
    }
    if (other_cpu->data_.size() != data_.size()) {
        throw std::runtime_error("CpuVector::copyFrom size mismatch");
    }
    data_ = other_cpu->data_;
    }

    void fill(double value) override {
        std::fill(data_.begin(), data_.end(), value);
    }

    size_t size() const override { return data_.size(); }

    // Internal access for CpuBackend
    std::vector<double>& getData() { return data_; }
    const std::vector<double>& getData() const { return data_; }
};

// --- CPU Sparse Matrix Implementation ---
class CpuSparseMatrix : public PdlpSparseMatrix {
    const HighsLp* lp_; // Just holds a reference
    HighsSparseMatrix matrix_col_wise_;

public:
    explicit CpuSparseMatrix(const HighsLp& lp) : lp_(&lp) {
        // Ensure we have a column-wise copy for Ax and ATy
        matrix_col_wise_ = lp.a_matrix_;
        matrix_col_wise_.ensureColwise();
    }

    size_t num_rows() const override { return matrix_col_wise_.num_row_; }
    size_t num_cols() const override { return matrix_col_wise_.num_col_; }

    // Internal access
    const HighsSparseMatrix& getMatrix() const { return matrix_col_wise_; }
    const HighsLp& getLp() const { return *lp_; }
};

// --- CPU Backend Implementation ---
class CpuBackend : public LinearAlgebraBackend {
private:
    // Helper to safely cast
    static CpuVector& as_cpu(PdlpVector& vec) {
        auto* p = dynamic_cast<CpuVector*>(&vec);
        if (!p) throw std::runtime_error("Invalid vector type for CpuBackend");
        return *p;
    }
    static const CpuVector& as_cpu(const PdlpVector& vec) {
        const auto* p = dynamic_cast<const CpuVector*>(&vec);
        if (!p) throw std::runtime_error("Invalid vector type for CpuBackend");
        return *p;
    }
    static const CpuSparseMatrix& as_cpu(const PdlpSparseMatrix& mat) {
        const auto* p = dynamic_cast<const CpuSparseMatrix*>(&mat);
        if (!p) throw std::runtime_error("Invalid matrix type for CpuBackend");
        return *p;
    }

public:
    std::unique_ptr<PdlpVector> createVector(size_t size) const override {
        return std::make_unique<CpuVector>(size);
    }

    std::unique_ptr<PdlpSparseMatrix> createSparseMatrix(
        const HighsLp& lp) const override {
        return std::make_unique<CpuSparseMatrix>(lp);
    }

    void Ax(PdlpVector& result, const PdlpSparseMatrix& A,
            const PdlpVector& x) const override {
        auto& res_vec = as_cpu(result).getData();
        const auto& mat = as_cpu(A).getMatrix();
        const auto& x_vec = as_cpu(x).getData();

        std::fill(res_vec.begin(), res_vec.end(), 0.0);
        for (HighsInt col = 0; col < mat.num_col_; ++col) {
            for (HighsInt i = mat.start_[col]; i < mat.start_[col + 1]; ++i) {
                const HighsInt row = mat.index_[i];
                res_vec[row] += mat.value_[i] * x_vec[col];
            }
        }
    }

    void ATy(PdlpVector& result, const PdlpSparseMatrix& A,
            const PdlpVector& y) const override {
        auto& res_vec = as_cpu(result).getData();
        const auto& mat = as_cpu(A).getMatrix();
        const auto& y_vec = as_cpu(y).getData();
        
        std::fill(res_vec.begin(), res_vec.end(), 0.0);
        for (HighsInt col = 0; col < mat.num_col_; ++col) {
            for (HighsInt i = mat.start_[col]; i < mat.start_[col + 1]; ++i) {
                const HighsInt row = mat.index_[i];
                res_vec[col] += mat.value_[i] * y_vec[row];
            }
        }
    }

    double dot(const PdlpVector& a, const PdlpVector& b) const override {
        const auto& a_vec = as_cpu(a).getData();
        const auto& b_vec = as_cpu(b).getData();
        double result = 0.0;
        for (size_t i = 0; i < a_vec.size(); ++i) {
            result += a_vec[i] * b_vec[i];
        }
        return result;
    }

    double nrm2(const PdlpVector& a) const override {
        return std::sqrt(dot(a, a));
    }

    double vector_norm(const PdlpVector& vec, double p) const override {
        const auto& v = as_cpu(vec).getData();
        if (std::isinf(p)) {
            double max_val = 0.0;
            for (double val : v) max_val = std::max(max_val, std::abs(val));
            return max_val;
        }
        if (p == 1.0) {
            double sum = 0.0;
            for (double val : v) sum += std::abs(val);
            return sum;
        }
        if (p == 2.0) {
            return nrm2(vec);
        }
        double sum = 0.0;
        for (double val : v) sum += std::pow(std::abs(val), p);
        return std::pow(sum, 1.0 / p);
    }


    void scale(PdlpVector& a, double factor) const override {
        auto& a_vec = as_cpu(a).getData();
        for (double& val : a_vec) {
        val *= factor;
        }
    }

    void sub(PdlpVector& result, const PdlpVector& a,
            const PdlpVector& b) const override {
        auto& res_vec = as_cpu(result).getData();
        const auto& a_vec = as_cpu(a).getData();
        const auto& b_vec = as_cpu(b).getData();
        for (size_t i = 0; i < res_vec.size(); ++i) {
        res_vec[i] = a_vec[i] - b_vec[i];
        }
    }

    void updateX(PdlpVector& x_new, const PdlpVector& x_old,
                const PdlpVector& aty, const PdlpVector& c,
                const PdlpVector& l, const PdlpVector& u,
                double primal_step) const override {
        auto& x_new_vec = as_cpu(x_new).getData();
        const auto& x_old_vec = as_cpu(x_old).getData();
        const auto& aty_vec = as_cpu(aty).getData();
        const auto& c_vec = as_cpu(c).getData();
        const auto& l_vec = as_cpu(l).getData();
        const auto& u_vec = as_cpu(u).getData();

        for (size_t i = 0; i < x_new_vec.size(); ++i) {
            double gradient = c_vec[i] - aty_vec[i];
            double x_unproj = x_old_vec[i] - primal_step * gradient;
            x_new_vec[i] = project_box(x_unproj, l_vec[i], u_vec[i]);
        }
    }

    void updateY(PdlpVector& y_new, const PdlpVector& y_old,
                const PdlpVector& ax, const PdlpVector& ax_next,
                const PdlpVector& rhs,
                const PdlpVector& is_equality_row_vec,
                double dual_step) const override {
        auto& y_new_vec = as_cpu(y_new).getData();
        const auto& y_old_vec = as_cpu(y_old).getData();
        const auto& ax_vec = as_cpu(ax).getData();
        const auto& ax_next_vec = as_cpu(ax_next).getData();
        const auto& rhs_vec = as_cpu(rhs).getData();
        const auto& is_eq_vec = as_cpu(is_equality_row_vec).getData();

        for (size_t j = 0; j < y_new_vec.size(); j++) {
            double extr_ax = 2 * ax_next_vec[j] - ax_vec[j];
            double dual_update = y_old_vec[j] + dual_step * (rhs_vec[j] - extr_ax);
            bool is_equality = (is_eq_vec[j] > 0.5);
            y_new_vec[j] =
                is_equality ? dual_update : project_non_negative(dual_update);
        }
    }

    void accumulate_weighted_sum(PdlpVector& sum_vec, const PdlpVector& vec,
                                double weight) const override {
        auto& s = as_cpu(sum_vec).getData();
        const auto& v = as_cpu(vec).getData();
        for (size_t i = 0; i < s.size(); ++i) {
            s[i] += v[i] * weight;
        }
    }

    void compute_average(PdlpVector& avg_vec, const PdlpVector& sum_vec,
                        double total_weight) const override {
        auto& avg = as_cpu(avg_vec).getData();
        const auto& sum = as_cpu(sum_vec).getData();
        const double scale = (total_weight > 1e-10) ? 1.0 / total_weight : 1.0;
        for (size_t i = 0; i < avg.size(); ++i) {
            avg[i] = sum[i] * scale;
        }
    }

    void computeDualSlacks(PdlpVector& dSlackPos, PdlpVector& dSlackNeg,
                            const PdlpVector& c, const PdlpVector& aty,
                            const PdlpVector& col_lower,
                            const PdlpVector& col_upper,
                            PdlpVector& dualResidual) const override {
        
        auto& s_pos = as_cpu(dSlackPos).getData();
        auto& s_neg = as_cpu(dSlackNeg).getData();
        const auto& c_vec = as_cpu(c).getData();
        const auto& aty_vec = as_cpu(aty).getData();
        const auto& l_vec = as_cpu(col_lower).getData();
        const auto& u_vec = as_cpu(col_upper).getData();
        auto& res_vec = as_cpu(dualResidual).getData();

        for (size_t i = 0; i < c_vec.size(); ++i) {
        res_vec[i] = c_vec[i] - aty_vec[i];

        if (l_vec[i] > -kHighsInf) {
            s_pos[i] = std::max(0.0, res_vec[i]);
        } else {
            s_pos[i] = 0.0;
        }
        if (u_vec[i] < kHighsInf) {
            s_neg[i] = std::max(0.0, -res_vec[i]);
        } else {
            s_neg[i] = 0.0;
        }
        }
    }

    double computePrimalFeasibility(
        const PdlpVector& ax, const PdlpVector& rhs,
        const PdlpVector& is_equality_row_vec,
        const std::vector<double>& row_scale,
        PdlpVector& primalResidual) const override {
        
        const auto& ax_vec = as_cpu(ax).getData();
        const auto& rhs_vec = as_cpu(rhs).getData();
        const auto& is_eq_vec = as_cpu(is_equality_row_vec).getData();
        auto& res_vec = as_cpu(primalResidual).getData();

        double norm_sq = 0.0;
        for (size_t i = 0; i < ax_vec.size(); ++i) {
        double residual = ax_vec[i] - rhs_vec[i];
        if (is_eq_vec[i] < 0.5) { // Inequality row
            residual = std::min(0.0, residual);
        }
        if (row_scale.size() > 0) {
            residual *= row_scale[i];
        }
        res_vec[i] = residual; // Store for debugging, not strictly needed
        norm_sq += residual * residual;
        }
        return std::sqrt(norm_sq);
    }

    double computeDualFeasibility(
        const PdlpVector& c, const PdlpVector& aty,
        const PdlpVector& dSlackPos, const PdlpVector& dSlackNeg,
        const std::vector<double>& col_scale,
        PdlpVector& dualResidual) const override {

        const auto& c_vec = as_cpu(c).getData();
        const auto& aty_vec = as_cpu(aty).getData();
        const auto& s_pos = as_cpu(dSlackPos).getData();
        const auto& s_neg = as_cpu(dSlackNeg).getData();
        auto& res_vec = as_cpu(dualResidual).getData();
        
        double norm_sq = 0.0;
        for (size_t i = 0; i < c_vec.size(); ++i) {
        double residual = (c_vec[i] - aty_vec[i]) - s_pos[i] + s_neg[i];
        if (col_scale.size() > 0) {
            residual *= col_scale[i];
        }
        res_vec[i] = residual; // Store for debugging
        norm_sq += residual * residual;
        }
        return std::sqrt(norm_sq);
    }

    void computeObjectives(
        const PdlpVector& c, const PdlpVector& x, const PdlpVector& rhs,
        const PdlpVector& y, const PdlpVector& col_lower,
        const PdlpVector& col_upper, const PdlpVector& dSlackPos,
        const PdlpVector& dSlackNeg, double offset, double& primal_obj,
        double& dual_obj) const override {
        
        // All dot products are simple host-side loops
        primal_obj = dot(c, x) + offset;
        
        double dual_obj_rhs = dot(rhs, y);
        double dual_obj_lower = dot(col_lower, dSlackPos);
        double dual_obj_upper = dot(col_upper, dSlackNeg);

        dual_obj = dual_obj_rhs + dual_obj_lower - dual_obj_upper + offset;
    }

    void synchronize() const override {
        // No-op for CPU
    }
};

// Factory function implementation
std::unique_ptr<LinearAlgebraBackend> createCpuBackend() {
  return std::make_unique<CpuBackend>();
}

}  // namespace pdlp
}  // namespace highs
#endif