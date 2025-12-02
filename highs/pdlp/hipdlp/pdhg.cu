#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> 

// Define Infinity for GPU 
#define GPU_INF 1e20 

// Buffer Indices for the reduction array
#define IDX_PRIMAL_FEAS 0
#define IDX_DUAL_FEAS   1
#define IDX_PRIMAL_OBJ  2
#define IDX_DUAL_OBJ    3

// Utility for robust 1D kernel launches
#define CUDA_GRID_STRIDE_LOOP(i, n)                                  \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// Helper function to calculate launch configuration
static dim3 GetLaunchConfig(int n, int block_size = 256) {
  int num_blocks = (n + block_size - 1) / block_size;
  return dim3(num_blocks, 1, 1);
}

// === KERNEL 1: Update X (Primal Step) ===
__global__ void kernelUpdateX(
    double* d_x_new, const double* d_x_old, const double* d_aty, 
    const double* d_cost, const double* d_lower, const double* d_upper,
    double primal_step, int n_cols) 
{
  CUDA_GRID_STRIDE_LOOP(i, n_cols) {
    // 1. Compute gradient: gradient = c - A'y
    double gradient = d_cost[i] - d_aty[i];
    
    // 2. Perform gradient step: x_updated = x_old - step * gradient
    double x_updated = d_x_old[i] - primal_step * gradient;
    
    // 3. Project to bounds [l, u]
    d_x_new[i] = fmax(d_lower[i], fmin(x_updated, d_upper[i]));
  }
}

// === KERNEL 2: Update Y (Dual Step) ===
__global__ void kernelUpdateY(
    double* d_y_new, const double* d_y_old,
    const double* d_ax_old, const double* d_ax_new,  
    const double* d_rhs, const bool* d_is_equality,
    double dual_step, int n_rows)
{
  CUDA_GRID_STRIDE_LOOP(j, n_rows) {
    double extra_ax = 2.0 * d_ax_new[j] - d_ax_old[j];
    double dual_update = d_y_old[j] + dual_step * (d_rhs[j] - extra_ax);
    if (d_is_equality[j]){// to be optimized 
      d_y_new[j] = dual_update;  // No bounds for equality constr aints
    } else {
      d_y_new[j] = fmax(0.0, dual_update);  // Project to non-negative orthant
    }
  }
}

// === KERNEL 3: Update Averages ===
// x_sum = x_sum + weight * x_next
// y_sum = y_sum + weight * y_next
__global__ void kernelUpdateAverages(
    double* d_x_sum, double* d_y_sum,
    const double* d_x_next, const double* d_y_next,
    double weight, int n_cols, int n_rows)
{
  CUDA_GRID_STRIDE_LOOP(i, n_cols) {
    d_x_sum[i] += weight * d_x_next[i];
  }
  CUDA_GRID_STRIDE_LOOP(j, n_rows) {
    d_y_sum[j] += weight * d_y_next[j];
  }
}

__global__ void kernelScaleVector(
    double* d_out, const double* d_in, 
    double scale, int n)
{
  CUDA_GRID_STRIDE_LOOP(i, n) {
    d_out[i] = d_in[i] * scale;
  }
}

// === KERNEL 4: Primal Convergence Check (Row-wise) ===
__global__ void kernelCheckPrimal(
  double* d_results,
  const double* d_ax, const double* d_y,
  const double* d_row_lower, const double* d_row_scale,
  const bool* d_is_equality, int n_rows){
  double local_feas_sq = 0.0;
  double local_dual_obj = 0.0;

  CUDA_GRID_STRIDE_LOOP(i,n_rows){
    double val_y = d_y[i];
    double val_b = d_row_lower[i];
    double val_ax = d_ax[i];

    if (abs(val_b) < GPU_INF){ 
      local_dual_obj += val_b * val_y;
    }

    double residual = val_ax - val_b;

    if (!d_is_equality[i]) {
      residual = fmin(0.0, residual);
    }

    if (d_row_scale != nullptr) {
      residual *= d_row_scale[i];
    }

    local_feas_sq += residual * residual;
  }

  // Atomic acculation
  // To be optimized: use warp-level reduction
  atomicAdd(&d_results[IDX_PRIMAL_FEAS], local_feas_sq);
  atomicAdd(&d_results[IDX_DUAL_OBJ], local_dual_obj);
}

// === KERNEL 5: Dual Convergence Check (Column-wise) ===
__global__ void kernelCheckDual(
    double* d_results,          // [1]: D.Feas, [2]: P.Obj, [3]: D.Obj
    double* d_slack_pos,        // Output
    double* d_slack_neg,        // Output
    const double* d_aty,
    const double* d_x,
    const double* d_cost,       // c
    const double* d_col_lower,  // l
    const double* d_col_upper,  // u
    const double* d_col_scale,  // Can be nullptr
    int n_cols)
{
  double local_dual_feas_sq = 0.0;
  double local_primal_obj = 0.0;
  double local_dual_obj_part = 0.0;

  CUDA_GRID_STRIDE_LOOP(i, n_cols){
    double val_x = d_x[i];
    double val_c = d_cost[i];
    double val_aty = d_aty[i];
    double val_l = d_col_lower[i];
    double val_u = d_col_upper[i];

    local_primal_obj += val_c * val_x;
    double dual_residual = val_c - val_aty;

    double s_pos = 0.0;
    double s_neg = 0.0;

    if (val_l > -GPU_INF) {
      s_pos = fmax(0.0, dual_residual);
    }

    if (val_u < GPU_INF) {
      s_neg = fmax(0.0, -dual_residual);
    }

    d_slack_pos[i] = s_pos;
    d_slack_neg[i] = s_neg;

    double eff_dual_residual = dual_residual - s_pos + s_neg;

    if (d_col_scale != nullptr) {
      eff_dual_residual *= d_col_scale[i];
    }

    local_dual_feas_sq += eff_dual_residual * eff_dual_residual;

    double obj_term = 0.0;
    if (val_l > -GPU_INF) obj_term += val_l * s_pos;
    if (val_u < GPU_INF) obj_term -= val_u * s_neg;

    local_dual_obj_part += obj_term;
  }

  // Atomic acculation
  atomicAdd(&d_results[IDX_DUAL_FEAS], local_dual_feas_sq);
  atomicAdd(&d_results[IDX_PRIMAL_OBJ], local_primal_obj);
  atomicAdd(&d_results[IDX_DUAL_OBJ], local_dual_obj_part);
}

__global__ void kernelDiffTwoNormSquared(
  const double* a, const double* b,
  double* result, int n){
  double local_diff_sq = 0.0;
  CUDA_GRID_STRIDE_LOOP(i, n){
    double diff = a[i] - b[i];
    local_diff_sq += diff * diff;
  }

  atomicAdd(result, local_diff_sq);
}

// Computes sum( (a_new[i] - a_old[i]) * (b_new[i] - b_old[i]) )
__global__ void kernelDiffDotDiff(
    const double* a_new, const double* a_old,
    const double* b_new, const double* b_old,
    double* result, int n) 
{
  double local_sum = 0.0;
  CUDA_GRID_STRIDE_LOOP(i, n) {
    double diff_a = a_new[i] - a_old[i];
    double diff_b = b_new[i] - b_old[i];
    local_sum += diff_a * diff_b;
  }
  atomicAdd(result, local_sum);
}

// Add C++ wrapper functions to launch the kernels
extern "C" {
void launchKernelUpdateX_wrapper(
    double* d_x_new, const double* d_x_old, const double* d_aty,
    const double* d_cost, const double* d_lower, const double* d_upper,
    double primal_step, int n_cols) 
{
    const int block_size = 256;
    dim3 config = GetLaunchConfig(n_cols, block_size);
    
    kernelUpdateX<<<config.x, block_size>>>(
        d_x_new, d_x_old, d_aty,
        d_cost, d_lower, d_upper,
        primal_step, n_cols);
    
    cudaGetLastError(); 
}

void launchKernelUpdateY_wrapper(
    double* d_y_new, const double* d_y_old,
    const double* d_ax_old, const double* d_ax_new, 
    const double* d_rhs, const bool* d_is_equality,
    double dual_step, int n_rows) 
{
    const int block_size = 256;
    dim3 config = GetLaunchConfig(n_rows, block_size);
    
    kernelUpdateY<<<config.x, block_size>>>(
        d_y_new, d_y_old,
        d_ax_old, d_ax_new,
        d_rhs, d_is_equality,
        dual_step, n_rows);
    
    cudaGetLastError();
}

void launchKernelUpdateAverages_wrapper(
    double* d_x_sum, double* d_y_sum,
    const double* d_x_next, const double* d_y_next,
    double weight, int n_cols, int n_rows) 
{
    const int block_size = 256;
    dim3 config_x = GetLaunchConfig(n_cols, block_size);
    dim3 config_y = GetLaunchConfig(n_rows, block_size);  
    kernelUpdateAverages<<<config_x.x > config_y.x ? config_x.x : config_y.x, block_size>>>(
        d_x_sum, d_y_sum,
        d_x_next, d_y_next,
        weight, n_cols, n_rows);
    cudaGetLastError();
}

void launchKernelScaleVector_wrapper(
    double* d_out, const double* d_in, 
    double scale, int n)
{
    const int block_size = 256;
    dim3 config = GetLaunchConfig(n, block_size);
    
    kernelScaleVector<<<config.x, block_size>>>(
        d_out, d_in, scale, n);
    
    cudaGetLastError();
}

void launchCheckConvergenceKernels_wrapper(
    double* d_results,
    double* d_slack_pos, double* d_slack_neg,
    const double* d_x, const double* d_y,
    const double* d_ax, const double* d_aty,
    const double* d_col_cost, const double* d_row_lower,
    const double* d_col_lower, const double* d_col_upper,
    const bool* d_is_equality,
    const double* d_col_scale, const double* d_row_scale,
    int n_cols, int n_rows)
{
    // 1. Zero out results
    cudaMemset(d_results, 0, 4 * sizeof(double));

    int block_size = 256;

    // 2. Launch Primal Kernel
    dim3 grid_rows = GetLaunchConfig(n_rows, block_size);
    kernelCheckPrimal<<<grid_rows, block_size>>>(
        d_results, d_ax, d_y, d_row_lower, d_row_scale, d_is_equality, n_rows
    );

    // 3. Launch Dual Kernel
    dim3 grid_cols = GetLaunchConfig(n_cols, block_size);
    kernelCheckDual<<<grid_cols, block_size>>>(
        d_results, d_slack_pos, d_slack_neg, d_aty, d_x, 
        d_col_cost, d_col_lower, d_col_upper, d_col_scale, n_cols
    );

    cudaGetLastError();
}

void launchKernelDiffTwoNormSquared_wrapper(
    const double* d_a, const double* d_b, double* d_result, int n) {
    
    // Reset result on device first
    cudaMemset(d_result, 0, sizeof(double));
    
    const int block_size = 256;
    dim3 config = GetLaunchConfig(n, block_size);
    kernelDiffTwoNormSquared<<<config.x, block_size>>>(d_a, d_b, d_result, n);
    cudaGetLastError();
}

void launchKernelDiffDotDiff_wrapper(
    const double* d_a_new, const double* d_a_old,
    const double* d_b_new, const double* d_b_old,
    double* d_result, int n) 
{
    cudaMemset(d_result, 0, sizeof(double));
    const int block_size = 256;
    dim3 config = GetLaunchConfig(n, block_size);
    
    kernelDiffDotDiff<<<config.x, block_size>>>(
        d_a_new, d_a_old, d_b_new, d_b_old, d_result, n);
    cudaGetLastError();
}
} // extern "C"