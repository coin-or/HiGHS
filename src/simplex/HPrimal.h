/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2019 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HPrimal.h
 * @brief Phase 2 primal simplex solver for HiGHS
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#ifndef SIMPLEX_HPRIMAL_H_
#define SIMPLEX_HPRIMAL_H_

#include "HConfig.h"
#include "lp_data/HighsModelObject.h"
#include "simplex/HSimplex.h"
#include "simplex/HVector.h"

/**
 * @brief Phase 2 primal simplex solver for HiGHS
 *
 * Not an efficient primal simplex solver: just a way of tidying up
 * dual infeasibilities when dual optimality (primal feasibility) has
 * been acheived with the dual simplex method
 */
class HPrimal {
 public:
  HPrimal(HighsModelObject& model_object) : workHMO(model_object) {}
  /**
   * @brief Solve a model instance
   */
  void solve();

  /**
   * @brief Perform Phase 2 primal simplex iterations
   */
  void solvePhase2();

 private:
  void primalRebuild();
  void primalChooseColumn();
  void primalChooseRow();
  void primalUpdate();

  void phase1ComputeDual();
  void phase1ChooseColumn();
  void phase1ChooseRow();
  void phase1Update();

  void devexReset();
  void devexUpdate();

  void iterationReport();
  void iterationReportFull(bool header);
  void iterationReportIterationAndPhase(int iterate_log_level, bool header);
  void iterationReportPrimalObjective(int iterate_log_level, bool header);
  void iterationReportIterationData(int iterate_log_level, bool header);
  void iterationReportRebuild(const int i_v);
  void reportInfeasibility();

  // Model pointer
  HighsModelObject& workHMO;

  int solver_num_col;
  int solver_num_row;
  int solver_num_tot;

  bool no_free_columns;

  int isPrimalPhase1;

  int solvePhase;
  int previous_iteration_report_header_iteration_count = -1;
  // Pivot related
  int invertHint;
  int columnIn;
  int rowOut;
  int columnOut;
  int phase1OutBnd;
  double thetaDual;
  double thetaPrimal;
  double alpha;
  //  double alphaRow;
  double numericalTrouble;
  int num_flip_since_rebuild;

  // Primal phase 1 tools
  vector<pair<double, int> > ph1SorterR;
  vector<pair<double, int> > ph1SorterT;

  // Devex weight
  int nBadDevexWeight;
  vector<double> devexWeight;
  vector<char> devexRefSet;

  // Solve buffer
  HVector row_ep;
  HVector row_ap;
  HVector column;

  double row_epDensity;
  double columnDensity;
};

#endif /* SIMPLEX_HPRIMAL_H_ */
