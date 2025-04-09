/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/raphael/Solver.cpp
 */
#include "Highs.h"
#include "pdlp/raphael/Solver.h"
#include "lp_data/HighsLpUtils.h"

HighsStatus solveLpRaphael(HighsLpSolverObject& solver_object) {
  return solveLpRaphael(solver_object.options_, solver_object.timer_,
			solver_object.lp_, solver_object.basis_,
			solver_object.solution_, solver_object.model_status_,
			solver_object.highs_info_, solver_object.callback_);
}

HighsStatus solveLpRaphael(const HighsOptions& options, HighsTimer& timer,
			   const HighsLp& lp, HighsBasis& highs_basis,
			   HighsSolution& highs_solution,
			   HighsModelStatus& model_status, HighsInfo& highs_info,
			   HighsCallback& callback) {
  // Indicate that there is no valid primal solution, dual solution or basis
  highs_basis.valid = false;
  highs_solution.value_valid = false;
  highs_solution.dual_valid = false;
  // Indicate that no imprecise solution has (yet) been found
  resetModelStatusAndHighsInfo(model_status, highs_info);

  double standard_form_offset;
  std::vector<double> standard_form_cost;
  std::vector<double> standard_form_rhs;
  HighsSparseMatrix standard_form_matrix;

  formStandardFormLp(lp,
		     options.log_options,
		     standard_form_offset,
		     standard_form_cost,
		     standard_form_rhs,
		     standard_form_matrix);
  const HighsInt num_col = standard_form_cost.size();
  const HighsInt num_row = standard_form_rhs.size();
  const HighsInt num_nz = standard_form_matrix.numNz();
  printf("Standard form LP has %d columns, %d rows and %d nonzeros\n",
	 int(num_col), int(num_row), int(num_nz));

  HighsModelStatus standard_form_model_status = HighsModelStatus::kNotset;
  double standard_form_objective_function_value = 0;
  HighsSolution standard_form_solution;

  HighsStatus status;
  const bool solve_with_simplex = true;
  if (solve_with_simplex) {
    status =
      solveStandardFormLpSimplex(options,
				 standard_form_offset,
				 standard_form_cost,
				 standard_form_rhs,
				 standard_form_matrix,
				 standard_form_model_status,
				 standard_form_objective_function_value,
				 standard_form_solution);
    if (status == HighsStatus::kError) return status;
  }
  // Now solve the LP in standard form using PDLP

  // Once solved, the solution for the LP in standard form obtained
  // with PDLP needs to be converted to a solution to the original
  // LP.
  standardFormSolutionToLpSolution(lp,
				   standard_form_solution,
				   highs_solution);
  // For the moment, return the model status as kSolveError, and HiGHS
  // status as error, so HiGHS doesn't expect anything in terms of a
  // primal or dual solution
  model_status = HighsModelStatus::kSolveError;
  return HighsStatus::kError;
}

HighsStatus solveStandardFormLpSimplex(const HighsOptions& options,
				       const double& standard_form_offset,
				       const std::vector<double>& standard_form_cost,
				       const std::vector<double>& standard_form_rhs,
				       const HighsSparseMatrix& standard_form_matrix,
				       HighsModelStatus& standard_form_model_status,
				       double& standard_form_objective_function_value,
				       HighsSolution& standard_form_solution) {
  Highs h;
  h.setOptionValue("output_flag", false);
  HighsLp lp;
  lp.num_col_ = standard_form_cost.size();
  lp.num_row_ = standard_form_rhs.size();
  lp.offset_ = standard_form_offset;
  lp.col_cost_ = standard_form_cost;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, kHighsInf);
  lp.row_lower_ = standard_form_rhs;
  lp.row_upper_ = standard_form_rhs;
  lp.a_matrix_ = standard_form_matrix;
  HighsStatus status = h.passModel(lp);
  if (status == HighsStatus::kError) return status;
  status = h.run();
  if (status == HighsStatus::kError) return status;
  standard_form_model_status = h.getModelStatus();
  standard_form_solution = h.getSolution();
  standard_form_objective_function_value = h.getInfo().objective_function_value;
  return status;
}
