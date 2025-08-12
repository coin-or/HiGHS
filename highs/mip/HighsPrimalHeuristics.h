/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_PRIMAL_HEURISTICS_H_
#define HIGHS_PRIMAL_HEURISTICS_H_

#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "util/HighsRandom.h"

class HighsMipSolver;

class HighsPrimalHeuristics {
 private:
  HighsMipSolver& mipsolver;
  size_t total_repair_lp;
  size_t total_repair_lp_feasible;
  size_t total_repair_lp_iterations;
  size_t lp_iterations;

  double successObservations;
  HighsInt numSuccessObservations;
  double infeasObservations;
  HighsInt numInfeasObservations;

  HighsRandom randgen;

  std::vector<HighsInt> intcols;

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);

  void setupIntCols();

  bool solveSubMip(const HighsLp& lp, const HighsBasis& basis,
                   double fixingRate, std::vector<double> colLower,
                   std::vector<double> colUpper, HighsInt maxleaves,
                   HighsInt maxnodes, HighsInt stallnodes);

  double determineTargetFixingRate();

  void rootReducedCost();

  void RENS(const std::vector<double>& relaxationsol);

  void RINS(const std::vector<double>& relaxationsol);

  void feasibilityPump();

  void centralRounding();

  void flushStatistics();

  bool tryRoundedPoint(const std::vector<double>& point,
                       const int solution_source);

  bool linesearchRounding(const std::vector<double>& point1,
                          const std::vector<double>& point2,
                          const int solution_source);

  void randomizedRounding(const std::vector<double>& relaxationsol);

  void shifting(const std::vector<double>& relaxationsol);

  void ziRound(const std::vector<double>& relaxationsol);

  HighsStatus solveMipKnapsackReturn(const HighsStatus& return_status);
  HighsStatus solveMipKnapsack();

  HighsStatus mipHeuristicInes();

  // #include "highs/Highs.h"
  // #include "highs/lp_data/HighsLpUtils.h"
  // #include <vector>
  // #include <algorithm>
  // #include <iostream>
  // #include <string>
  // #include<cmath>
  // #include <typeinfo>
  //
  // using namespace std;
};

void simpleOnlineAlgo(vector<HighsInt>& solution, HighsSparseMatrix& AMatrix,
                      vector<double> bVector, vector<double> rVector,
                      HighsInt checkConstraints, HighsInt minOrMax,
                      vector<HighsInt>& order, int checkR, HighsInt goingBack);
void dotProduct(const vector<HighsInt>& index, const vector<double>& value,
                const vector<double>& p, const int nonZero, double& sum);
HighsInt checkZeroFeasible(const vector<double> bVector, int nConstraints,
                           Highs& highs);
HighsInt checkFeasibility(const vector<HighsInt>& solution,
                          HighsSparseMatrix& AMatrix,
                          const vector<double>& bVector, int nRows);
void treatMatrices(HighsSparseMatrix& AMatrix, vector<double>& bVector,
                   vector<double>& rVector, HighsInt problemForm, int& numRow,
                   Highs& highs);
void changeRow(HighsSparseMatrix& AMatrix, vector<double>& bVector,
               HighsInt mult, HighsInt rowIndex, vector<double>& bPrimeVector);
void sortOrder(HighsSparseMatrix& AMatrix, vector<double>& rVector,
               HighsInt nVariables, vector<HighsInt>& indices,
               HighsInt sortNumber, vector<double>& bVector, HighsInt nRows);
void updatePt(vector<double>& p, const HighsInt nonZero,
              const vector<HighsInt>& index, const vector<double>& value,
              double gamma, HighsInt xt, HighsInt nRows, HighsInt nVariables,
              vector<double> bVector);
void runAndPrintAlgo(Highs& highs, HighsSparseMatrix& AMatrix,
                     vector<double>& bVector, vector<double>& rVector,
                     HighsInt algoNumber, HighsInt minMax,
                     vector<HighsInt>& indices, HighsInt CheckR,
                     const string& label, HighsInt goingBack);
void compareWithHighsSol(Highs& highs);
HighsInt computeSumAXi(HighsInt start, HighsInt end, HighsSparseMatrix& AMatrix,
                       vector<HighsInt>& solution, vector<HighsInt>& order);
HighsInt findIndex(const vector<HighsInt>& order, HighsInt val);
void printColiOfA(HighsSparseMatrix& AMatrix, HighsInt i);
HighsInt computeSumRFromiToj(vector<double>& rVector, vector<HighsInt>& order,
                             HighsInt start, HighsInt end,
                             vector<HighsInt>& solution);
void swapXis(vector<HighsInt>& solution, HighsInt indexJ, HighsInt indexT,
             vector<HighsInt>& order);
void runGoingBackAlgo(vector<double>& constraintsChecker, HighsInt t,
                      vector<double>& rVector, vector<HighsInt>& order,
                      vector<HighsInt>& solution, HighsSparseMatrix& AMatrix,
                      vector<double> bVector, HighsInt nConstraints);
HighsInt checkConstraintsForSwap(HighsInt indexJ, HighsInt indexT,
                                 HighsSparseMatrix& AMatrix,
                                 vector<double>& bVector, HighsInt nConstraints,
                                 vector<HighsInt>& order,
                                 vector<HighsInt>& solution);
void changeProblemMatrices(HighsSparseMatrix& AMatrix, vector<double>& bVector,
                           vector<double>& rVector, Highs& highs);
void checkFeasibilityStepT(vector<double>& constraintsChecker,
                           HighsSparseMatrix& AMatrix, vector<double>& bVector,
                           HighsInt stepT, HighsInt& constraintsNotRespected);
void updateConstraintsCheckBeforeSwap(vector<double>& constraintsChecker,
                                      HighsSparseMatrix& AMatrix,
                                      vector<HighsInt>& solution,
                                      HighsInt indexJ, HighsInt indexT,
                                      vector<HighsInt>& order);
#endif
