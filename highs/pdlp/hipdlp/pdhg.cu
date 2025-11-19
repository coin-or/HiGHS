#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> 

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
} // extern "C"