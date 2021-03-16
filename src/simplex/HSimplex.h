/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HSimplex.h
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#ifndef SIMPLEX_HSIMPLEX_H_
#define SIMPLEX_HSIMPLEX_H_

#include "lp_data/HighsModelObject.h"

enum class LpAction {
  SCALE = 0,
  NEW_COSTS,
  NEW_BOUNDS,
  NEW_BASIS,
  NEW_COLS,
  NEW_ROWS,
  DEL_COLS,
  DEL_ROWS,
  DEL_ROWS_BASIS_OK,
  SCALED_COL,
  SCALED_ROW,
  BACKTRACKING
};

void scaleAndPassLpToEkk(HighsModelObject& highs_model_object);

void choosePriceTechnique(const int price_strategy, const double row_ep_density,
                          bool& use_col_price, bool& use_row_price_w_switch);

void appendNonbasicColsToBasis(HighsLp& lp, HighsBasis& basis, int XnumNewCol);
void appendNonbasicColsToBasis(HighsLp& lp, SimplexBasis& basis,
                               int XnumNewCol);

void appendBasicRowsToBasis(HighsLp& lp, HighsBasis& basis, int XnumNewRow);
void appendBasicRowsToBasis(HighsLp& lp, SimplexBasis& basis, int XnumNewRow);

void invalidateSimplexLpBasisArtifacts(
    HighsSimplexLpStatus&
        simplex_lp_status  // !< Status of simplex LP whose
                           // basis artifacts are to be invalidated
);

void invalidateSimplexLpBasis(
    HighsSimplexLpStatus& simplex_lp_status  // !< Status of simplex LP whose
                                             // basis is to be invalidated
);

void invalidateSimplexLp(
    HighsSimplexLpStatus&
        simplex_lp_status  // !< Status of simplex LP to be invalidated
);

void updateSimplexLpStatus(
    HighsSimplexLpStatus&
        simplex_lp_status,  // !< Status of simplex LP to be updated
    LpAction action         // !< Action prompting update
);

void unscaleSolution(HighsSolution& solution, const HighsScale scale);

HighsStatus deleteScale(const HighsLogOptions& log_options,
                        vector<double>& scale,
                        const HighsIndexCollection& index_collection);

void getUnscaledInfeasibilitiesAndNewTolerances(
    const HighsOptions& options, const HighsLp& lp,
    const HighsModelStatus model_status, const SimplexBasis& basis,
    const HighsSimplexInfo& simplex_info, const HighsScale& scale,
    HighsSolutionParams& solution_params,
    double& new_primal_feasibility_tolerance,
    double& new_dual_feasibility_tolerance);

// SCALE:

// void initialiseScale(HighsModelObject& highs_model);

void initialiseScale(const HighsLp& lp, HighsScale& scale);

void scaleSimplexLp(const HighsOptions& options, HighsLp& lp,
                    HighsScale& scale);
void scaleCosts(const HighsOptions& options, HighsLp& lp, double& cost_scale);
bool equilibrationScaleSimplexMatrix(const HighsOptions& options, HighsLp& lp,
                                     HighsScale& scale);
bool maxValueScaleSimplexMatrix(const HighsOptions& options, HighsLp& lp,
                                HighsScale& scale);

bool isBasisRightSize(const HighsLp& lp, const SimplexBasis& basis);

/*
void computeDualObjectiveValue(HighsModelObject& highs_model_object, int phase =
2); void computePrimalObjectiveValue(HighsModelObject& highs_model_object);
double computeBasisCondition(const HighsModelObject& highs_model_object);
*/
#endif  // SIMPLEX_HSIMPLEX_H_
