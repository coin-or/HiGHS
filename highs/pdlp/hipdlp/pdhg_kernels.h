#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launchKernelUpdateX_wrapper(
    double* d_x_new, const double* d_x_old, const double* d_aty,
    const double* d_cost, const double* d_lower, const double* d_upper,
    double primal_step, int n_cols);

void launchKernelUpdateY_wrapper(
    double* d_y_new, const double* d_y_old,
    const double* d_ax_old, const double* d_ax_new, 
    const double* d_row_lower, const bool* d_is_equality,
    double dual_step, int n_rows);

void launchKernelUpdateAverages_wrapper(
    double* d_x_sum, double* d_y_sum,
    const double* d_x_current, const double* d_y_current,
    double weight, int n_cols, int n_rows);

void launchKernelScaleVector_wrapper(
    double* d_out, const double* d_in, 
    double scale, int n);

#ifdef __cplusplus
}
#endif