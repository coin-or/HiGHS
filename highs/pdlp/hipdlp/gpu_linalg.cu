/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/hipdlp/gpu_linalg.cu
 * @brief GPU implementation of the linear algebra interface.
 */
#include "gpu_linalg.hpp"
#include "cpu_linalg.hpp" // For fallback factory function
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      throw std::runtime_error(std::string("CUDA Error: ") +               \
                               cudaGetErrorString(err) + " at " +         \
                               std::string(__FILE__) + ":" +              \
                               std::to_string(__LINE__));                 \
    }                                                                     \
  } while (0)

#define CUSPARSE_CHECK(call)                                              \
  do {                                                                    \
    cusparseStatus_t status = call;                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                              \
      throw std::runtime_error(std::string("cuSPARSE Error: ") +           \
                               cusparseGetErrorString(status) + " at " +  \
                               std::string(__FILE__) + ":" +              \
                               std::to_string(__LINE__));                 \
    }                                                                     \
  } while (0)

#define CUBLAS_CHECK(call) /* (Omitted for brevity, same as before) */     \
  do {                                                                    \
    cublasStatus_t status = call;                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      throw std::runtime_error(std::string("cuBLAS Error"));              \
    }                                                                     \
  } while (0)

namespace highs {
namespace pdlp {    

// --- CUDA Kernels ---

__global__ void updateX_kernel(size_t n, double* __restrict__ x_new,
                               const double* __restrict__ x_old,
                               const double* __restrict__ aty,
                               const double* __restrict__ c,
                               const double* __restrict__ l,
                               const double* __restrict__ u,
                               double primal_step) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;    
    if (i >= n) return;
    double gradient = c[i] - aty[i];
    double x_unproj = x_old[i] - primal_step * gradient;
    x_new[i] = fmax(l[i], fmin(u[i], x_unproj));
}

__global__ void updateY_kernel(size_t n, double* __restrict__ y_new,
                               const double* __restrict__ y_old,
                               const double* __restrict__ ax,
                               const double* __restrict__ ax_next,
                               const double* __restrict__ rhs,
                               const double* __restrict__ is_equality_row_vec,
                               double dual_step) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;

  double extr_ax = 2 * ax_next[j] - ax[j];
  double dual_update = y_old[j] + dual_step * (rhs[j] - extr_ax);
  bool is_equality = (is_equality_row_vec[j] > 0.5);
  y_new[j] = is_equality ? dual_update : fmax(0.0, dual_update);
}

__global__ void accumulate_weighted_sum_kernel(size_t n, double* __restrict__ sum_vec,
                                               const double* __restrict__ vec,
                                               double weight) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sum_vec[i] += vec[i] * weight;
}

__global__ void compute_average_kernel(size_t n, double* __restrict__ avg_vec,
                                       const double* __restrict__ sum_vec,
                                       double scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    avg_vec[i] = sum_vec[i] * scale;
}

__global__ void fill_kernel(size_t n, double* __restrict__ vec, double value) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vec[i] = value;
}

__global__ void sub_kernel(size_t n, double* __restrict__ result, 
                           const double* __restrict__ a,
                           const double* __restrict__ b) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    result[i] = a[i] - b[i];
}

__global__ void computeDualSlacks_kernel(
    size_t n, double* __restrict__ dSlackPos, double* __restrict__ dSlackNeg,
    const double* __restrict__ c, const double* __restrict__ aty,
    const double* __restrict__ col_lower,
    const double* __restrict__ col_upper,
    double* __restrict__ dualResidual) {
  
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double res = c[i] - aty[i];
  dualResidual[i] = res; // Store temp result

  if (col_lower[i] > -kHighsInf) {
    dSlackPos[i] = fmax(0.0, res);
  } else {
    dSlackPos[i] = 0.0;
  }
  
  if (col_upper[i] < kHighsInf) {
    dSlackNeg[i] = fmax(0.0, -res);
  } else {
    dSlackNeg[i] = 0.0;
  }
}

__global__ void computePrimalResidual_kernel(
    size_t n, const double* __restrict__ ax, const double* __restrict__ rhs,
    const double* __restrict__ is_equality_row_vec,
    const double* __restrict__ row_scale, size_t row_scale_size,
    double* __restrict__ primalResidual) {

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double residual = ax[i] - rhs[i];
  if (is_equality_row_vec[i] < 0.5) { // Inequality row
    residual = fmin(0.0, residual);
  }
  
  if (row_scale_size > 0) {
    residual *= row_scale[i];
  }
  
  primalResidual[i] = residual;
}


__global__ void computeDualResidual_kernel(
    size_t n, const double* __restrict__ c, const double* __restrict__ aty,
    const double* __restrict__ dSlackPos,
    const double* __restrict__ dSlackNeg,
    const double* __restrict__ col_scale, size_t col_scale_size,
    double* __restrict__ dualResidual) {

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  
  double residual = (c[i] - aty[i]) - dSlackPos[i] + dSlackNeg[i];
  
  if (col_scale_size > 0) {
    residual *= col_scale[i];
  }
  
  dualResidual[i] = residual;
}

// --- GPU Vector Implementation ---
class GpuVector : public PdlpVector {
    double* d_data_ = nullptr;
    size_t size_ = 0;

public:
    explicit GpuVector(size_t size) : size_(size) {
        if (size_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_data_, size_ * sizeof(double)));
        }
    }

    ~GpuVector() override {
        if (d_data_) {
        cudaFree(d_data_); // Note: CUDA_CHECK in dtor is risky
        }
    }

    // Disable copy/move
    GpuVector(const GpuVector&) = delete;
    GpuVector& operator=(const GpuVector&) = delete;

    void copyFromHost(const std::vector<double>& host_data) override {
        if (host_data.size() != size_) {
            throw std::runtime_error("Host data size does not match vector size.");
        }
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_data_, host_data.data(), size_ * sizeof(double),
                                cudaMemcpyHostToDevice));
        }
    }

    void copyToHost(std::vector<double>& host_data) const override {
        if (host_data.size() != size_) {
            host_data.resize(size_);
        }
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(host_data.data(), d_data_, size_ * sizeof(double),
                                cudaMemcpyDeviceToHost));
        }
    }

    void copyFrom(const PdlpVector& other) override {
        const auto* other_gpu = dynamic_cast<const GpuVector*>(&other);
        if (!other_gpu) {
            throw std::runtime_error("Cannot copy from non-GPU vector to GPU vector");
        }
        if (other_gpu->size_ != size_) {
            throw std::runtime_error("GpuVector::copyFrom size mismatch");
        }
        if (size_ > 0) {
            CUDA_CHECK(cudaMemcpy(d_data_, other_gpu->d_data_, size_ * sizeof(double),
                                cudaMemcpyDeviceToDevice));
        }
    }

    void fill(double value) override {
        if (size_ == 0) return;
        dim3 block(256);
        dim3 grid((size_ + block.x - 1) / block.x);
        fill_kernel<<<grid, block>>>(size_, d_data_, value);
        CUDA_CHECK(cudaGetLastError());
    }

    size_t size() const override { return size_; }

    // Internal access
    double* deviceData() { return d_data_; }
    const double* deviceData() const { return d_data_; }
};

// --- GPU Sparse Matrix Implementation ---
class GpuSparseMatrix : public PdlpSparseMatrix {
    size_t num_rows_ = 0;
    size_t num_cols_ = 0;
    size_t nnz_ = 0;

    // cuSPARSE descriptors
    cusparseSpMatDescr_t mat_descr_ = nullptr;
    
    // Device data pointers (CSC format)
    void* d_csc_col_ptr_ = nullptr;   // size = num_cols + 1
    void* d_csc_row_ind_ = nullptr;   // size = nnz
    void* d_values_      = nullptr;   // size = nnz

public:
    explicit GpuSparseMatrix(const HighsLp& lp, cusparseHandle_t handle)
        : num_rows_(lp.num_row_), num_cols_(lp.num_col_), nnz_(lp.a_matrix_.numNz()){
        // 1. Ensure matrix is column-wise (CSC) on host
        HighsSparseMatrix csc_matrix = lp.a_matrix_;
        csc_matrix.ensureColwise();

        // 2. Allocate device arrays (CSC format)
        CUDA_CHECK(cudaMalloc(&d_csc_col_ptr_, (num_cols_ + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_csc_row_ind_, nnz_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values_, nnz_ * sizeof(double)));

        // 3. Copy data to device
        CUDA_CHECK(cudaMemcpy(d_csc_col_ptr_, csc_matrix.start_.data(),
                            (num_cols_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csc_row_ind_, csc_matrix.index_.data(),
                            nnz_ * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values_, csc_matrix.value_.data(),
                            nnz_ * sizeof(double), cudaMemcpyHostToDevice));

        // 4. Create cuSPARSE matrix descriptor
        CUSPARSE_CHECK(cusparseCreateCsc(&mat_descr_, num_rows_, num_cols_, nnz_,
                                        d_csc_col_ptr_, d_csc_row_ind_, d_values_,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, // row ptr type, col ind type
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)); // index base, value type
    }

    ~GpuSparseMatrix() override {
        if(mat_descr_) cusparseDestroySpMat(mat_descr_);
        if(d_csc_col_ptr_) cudaFree(d_csc_col_ptr_);
        if(d_csc_row_ind_) cudaFree(d_csc_row_ind_);
        if(d_values_) cudaFree(d_values_);  
    }

    // Disable copy/move
    GpuSparseMatrix(const GpuSparseMatrix&) = delete;
    GpuSparseMatrix& operator=(const GpuSparseMatrix&) = delete;

    size_t num_rows() const override { return num_rows_; }
    size_t num_cols() const override { return num_cols_; }

    // Internal access
    cusparseSpMatDescr_t getMatrixDescr() const { return mat_descr_; }
};

// --- GPU Backend Implementation ---
class GpuBackend : public LinearAlgebraBackend {
    // --- Handles ---
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    mutable void* d_buffer_ = nullptr; // Persistent buffer for SpMV
    mutable size_t buffer_size_ = 0;
    
    // Helper to safely cast
    static GpuVector& as_gpu(PdlpVector& vec) {
        auto* p = dynamic_cast<GpuVector*>(&vec);
        if (!p) throw std::runtime_error("Invalid vector type for GpuBackend");
        return *p;
    }
    static const GpuVector& as_gpu(const PdlpVector& vec) {
        const auto* p = dynamic_cast<const GpuVector*>(&vec);
        if (!p) throw std::runtime_error("Invalid vector type for GpuBackend");
        return *p;
    }
    static const GpuSparseMatrix& as_gpu(const PdlpSparseMatrix& mat) {
        const auto* p = dynamic_cast<const GpuSparseMatrix*>(&mat);
        if (!p) throw std::runtime_error("Invalid matrix type for GpuBackend");
        return *p;
    }
    
    // Helper to get raw device pointer
    static double* d_ptr(PdlpVector& vec) { return as_gpu(vec).deviceData(); }
    static const double* d_ptr(const PdlpVector& vec) { return as_gpu(vec).deviceData(); }

    // Host-side buffer for scaling vectors
    double* d_row_scale_ = nullptr;
    size_t row_scale_size_ = 0;
    double* d_col_scale_ = nullptr;
    size_t col_scale_size_ = 0;

public:
    GpuBackend() {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }

    ~GpuBackend() override {
        if (d_buffer_) cudaFree(d_buffer_);
        cublasDestroy(cublas_handle_);
        cusparseDestroy(cusparse_handle_);
    }

    std::unique_ptr<PdlpVector> createVector(size_t size) const override {
        return std::make_unique<GpuVector>(size);
    }

    std::unique_ptr<PdlpSparseMatrix> createSparseMatrix(
        const HighsLp& lp) const override {
        return std::make_unique<GpuSparseMatrix>(lp, cusparse_handle_);
    }

    // --- Core Linear Algebra (SpMV) ---
    void Ax(PdlpVector& result, const PdlpSparseMatrix& A,
          const PdlpVector& x) const override {
        const auto& mat = as_gpu(A);
        cusparseDnVecDescr_t vec_x, vec_y;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, mat.num_cols(), (void*)d_ptr(x), CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, mat.num_rows(), d_ptr(result), CUDA_R_64F));
        const double alpha = 1.0, beta = 0.0;
        allocateSpMvBuffer(CUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, vec_x, beta, vec_y);
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat.getMatrixDescr(), vec_x, &beta, vec_y, CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2, d_buffer_)); // <-- Use ALG2 as requested
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    }
  
    void ATy(PdlpVector& result, const PdlpSparseMatrix& A,
            const PdlpVector& y) const override {
        const auto& mat = as_gpu(A);
        cusparseDnVecDescr_t vec_y, vec_x;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, mat.num_rows(), (void*)d_ptr(y), CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, mat.num_cols(), d_ptr(result), CUDA_R_64F));
        const double alpha = 1.0, beta = 0.0;
        allocateSpMvBuffer(CUSPARSE_OPERATION_TRANSPOSE, alpha, mat, vec_y, beta, vec_x);
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle_, CUSPARSE_OPERATION_TRANSPOSE,
            &alpha, mat.getMatrixDescr(), vec_y, &beta, vec_x, CUDA_R_64F,
            CUSPARSE_SPMV_CSR_ALG2, d_buffer_)); // <-- Use ALG2 as requested
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    }

    // --- BLAS-like Operations ---
    double dot(const PdlpVector& a, const PdlpVector& b) const override {
        if (a.size() != b.size()) {
            throw std::runtime_error("GpuBackend::dot size mismatch");
        }
        double result = 0.0;
        CUBLAS_CHECK(cublasDdot(cublas_handle_, a.size(), d_ptr(a), 1, d_ptr(b), 1, &result));
        return result;
    }

    double nrm2(const PdlpVector& a) const override {
        if (a.size() == 0) return 0.0;
        double result = 0.0;
        CUBLAS_CHECK(cublasDnrm2(cublas_handle_, a.size(), d_ptr(a), 1, &result));
        return result;
    }
    
    double vector_norm(const PdlpVector& vec, double p) const override {
        // cuBLAS only has nrm2. For inf and 1-norm, we'd need custom kernels or
        // cublasDasum/cublasIdamax (which return indices).
        // For simplicity, we'll just implement nrm2 (p=2.0).
        if (p == 2.0) {
            return nrm2(vec);
        }
        
        if (std::isinf(p)) {
            int index = 0;
            // cuBLAS finds the *index* of the max element
            CUBLAS_CHECK(cublasIdamax(cublas_handle_, vec.size(), d_ptr(vec), 1, &index));
            
            // cublasIdamax is 1-based, convert to 0-based
            index = index - 1; 
            
            // We must copy just this one element back
            double result = 0.0;
            CUDA_CHECK(cudaMemcpy(&result, d_ptr(vec) + index, sizeof(double), cudaMemcpyDeviceToHost));
            return std::abs(result);
        }
        if (p == 1.0) {
            double result = 0.0;
            CUBLAS_CHECK(cublasDasum(cublas_handle_, vec.size(), d_ptr(vec), 1, &result));
            return result;
        }

        // Fallback for other p-norms (slow)
        std::vector<double> h_vec;
        vec.copyToHost(h_vec);
        double sum = 0.0;
        for (double val : h_vec) sum += std::pow(std::abs(val), p);
        return std::pow(sum, 1.0 / p);
    }

    void scale(PdlpVector& a, double factor) const override {
        if (a.size() == 0) return;
        CUBLAS_CHECK(cublasDscal(cublas_handle_, a.size(), &factor, d_ptr(a), 1));
    }

    void sub(PdlpVector& result, const PdlpVector& a,
            const PdlpVector& b) const override {
        if (result.size() == 0) return;
        size_t n = result.size();
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        sub_kernel<<<grid, block>>>(n, d_ptr(result), d_ptr(a), d_ptr(b));
        CUDA_CHECK(cudaGetLastError());
    }

    // PDHG Kernels
    void updateX(PdlpVector& x_new, const PdlpVector& x_old,
                const PdlpVector& aty, const PdlpVector& c,
                const PdlpVector& l, const PdlpVector& u,
                double primal_step) const override {
        size_t n = x_new.size();
        if (n == 0) return;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        updateX_kernel<<<grid, block>>>(n, d_ptr(x_new), d_ptr(x_old), d_ptr(aty),
                                        d_ptr(c), d_ptr(l), d_ptr(u), primal_step);
        CUDA_CHECK(cudaGetLastError());
    }

    void updateY(PdlpVector& y_new, const PdlpVector& y_old,
                const PdlpVector& ax, const PdlpVector& ax_next,
                const PdlpVector& rhs,
                const PdlpVector& is_equality_row_vec,
                double dual_step) const override {
        size_t n = y_new.size();
        if (n == 0) return;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        updateY_kernel<<<grid, block>>>(n, d_ptr(y_new), d_ptr(y_old), d_ptr(ax),
                                        d_ptr(ax_next), d_ptr(rhs),
                                        d_ptr(is_equality_row_vec), dual_step);
        CUDA_CHECK(cudaGetLastError());
    }

    // Averaging
    void accumulate_weighted_sum(PdlpVector& sum_vec, const PdlpVector& vec,
                                double weight) const override {
        size_t n = sum_vec.size();
        if (n == 0) return;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        accumulate_weighted_sum_kernel<<<grid, block>>>(n, d_ptr(sum_vec), d_ptr(vec), weight);
        CUDA_CHECK(cudaGetLastError());
    }

    void compute_average(PdlpVector& avg_vec, const PdlpVector& sum_vec,
                        double total_weight) const override {
        size_t n = avg_vec.size();
        if (n == 0) return;
        const double scale = (total_weight > 1e-10) ? 1.0 / total_weight : 1.0;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        compute_average_kernel<<<grid, block>>>(n, d_ptr(avg_vec), d_ptr(sum_vec), scale);
        CUDA_CHECK(cudaGetLastError());
    }

    void computeDualSlacks(PdlpVector& dSlackPos, PdlpVector& dSlackNeg,
                         const PdlpVector& c, const PdlpVector& aty,
                         const PdlpVector& col_lower,
                         const PdlpVector& col_upper,
                         PdlpVector& dualResidual) const override {
        size_t n = c.size();
        if (n == 0) return;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        computeDualSlacks_kernel<<<grid, block>>>(
            n, d_ptr(dSlackPos), d_ptr(dSlackNeg), d_ptr(c), d_ptr(aty),
            d_ptr(col_lower), d_ptr(col_upper), d_ptr(dualResidual));
        CUDA_CHECK(cudaGetLastError());
    }

    double computePrimalFeasibility(
      const PdlpVector& ax, const PdlpVector& rhs,
      const PdlpVector& is_equality_row_vec,
      const std::vector<double>& row_scale,
      PdlpVector& primalResidual) const override {
    
        // --- Upload scaling vector (if changed) ---
        if (row_scale_size_ != row_scale.size()) {
            if(d_row_scale_) cudaFree(d_row_scale_);
            row_scale_size_ = row_scale.size();
            if (row_scale_size_ > 0) {
                CUDA_CHECK(cudaMalloc(&d_row_scale_, row_scale_size_ * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_row_scale_, row_scale.data(), row_scale_size_ * sizeof(double), cudaMemcpyHostToDevice));
            }
        }

        size_t n = ax.size();
        if (n == 0) return 0.0;
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        computePrimalResidual_kernel<<<grid, block>>>(
            n, d_ptr(ax), d_ptr(rhs), d_ptr(is_equality_row_vec),
            d_row_scale_, row_scale_size_, d_ptr(primalResidual));
        CUDA_CHECK(cudaGetLastError());
        
        // Compute L2 norm of the result
        return nrm2(primalResidual);
    }


    void synchronize() const override {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

private:
    // Mutable member function to manage buffer allocation
    void allocateSpMvBuffer(cusparseOperation_t op, double alpha, const GpuSparseMatrix& mat,
                          const cusparseDnVecDescr_t vec_in, double beta,
                          const cusparseDnVecDescr_t vec_out) const{
        size_t new_size = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse_handle_, op, &alpha, mat.getMatrixDescr(), vec_in,
            &beta, vec_out, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &new_size)); // cupldp Use ALG2 for CSR
        if (new_size > buffer_size_) {
        if (d_buffer_) CUDA_CHECK(cudaFree(d_buffer_));
        buffer_size_ = new_size;
        CUDA_CHECK(cudaMalloc(&d_buffer_, buffer_size_));
        }
    }
};

// Factory function implementation
std::unique_ptr<LinearAlgebraBackend> createGpuBackend() {
  // Check if GPU is available
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
      std::cerr << "Warning: GPU backend requested, but no CUDA-capable device found. "
                << "Falling back to CPU." << std::endl;
      return createCpuBackend();
  }
  return std::make_unique<GpuBackend>();
}

}
}
