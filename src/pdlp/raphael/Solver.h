/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file pdlp/raphael/Solver.h
 * @brief
 */
#ifndef PDLP_RAPHAEL_SOLVER_H_
#define PDLP_RAPHAEL_SOLVER_H_

#include <algorithm>
#include <cassert>

#include "lp_data/HighsSolution.h"

HighsStatus solveLpRaphael(HighsLpSolverObject& solver_object);
HighsStatus solveLpRaphael(const HighsOptions& options, HighsTimer& timer,
			   const HighsLp& lp, HighsBasis& highs_basis,
			   HighsSolution& highs_solution,
			   HighsModelStatus& model_status, HighsInfo& highs_info,
			   HighsCallback& callback);

HighsStatus solveStandardFormLpSimplex(const HighsOptions& options,
				       const double& standard_form_offset,
				       const std::vector<double>& standard_form_cost,
				       const std::vector<double>& standard_form_rhs,
				       const HighsSparseMatrix& standard_form_matrix,
				       HighsModelStatus& standard_form_model_status,
				       double& standard_form_objective_function_value,
				       HighsSolution& standard_form_solution);
#endif
