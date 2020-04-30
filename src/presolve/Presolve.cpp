/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2020 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file presolve/Presolve.cpp
 * @brief
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#include "presolve/Presolve.h"

#include "io/HighsIO.h"
#include "lp_data/HConst.h"

//#include "simplex/HFactor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <queue>
#include <sstream>

#include "test/KktChStep.h"

namespace presolve {

using std::cout;
using std::endl;
using std::flush;
using std::get;
using std::ios;
using std::list;
using std::make_pair;
using std::max;
using std::min;
using std::ofstream;
using std::setprecision;
using std::setw;
using std::stringstream;

constexpr int iPrint = -1;
// todo:
// iKKTcheck = 1;

void Presolve::load(const HighsLp& lp) {
  timer.recordStart(MATRIX_COPY);
  numCol = lp.numCol_;
  numRow = lp.numRow_;
  numTot = numTot;
  Astart = lp.Astart_;
  Aindex = lp.Aindex_;
  Avalue = lp.Avalue_;

  colCost = lp.colCost_;
  if (lp.sense_ == ObjSense::MAXIMIZE) {
    for (unsigned int col = 0; col < lp.colCost_.size(); col++)
      colCost[col] = -colCost[col];
  }

  colLower = lp.colLower_;
  colUpper = lp.colUpper_;
  rowLower = lp.rowLower_;
  rowUpper = lp.rowUpper_;

  modelName = lp.model_name_;
  timer.recordFinish(MATRIX_COPY);
}

void Presolve::setBasisInfo(
    const std::vector<HighsBasisStatus>& pass_col_status,
    const std::vector<HighsBasisStatus>& pass_row_status) {
  col_status = pass_col_status;
  row_status = pass_row_status;
}

// printing with cout goes here.
void reportDev(const string& message) {
  if (iPrint == -1) std::cout << message << std::flush;
  return;
}

// todo:
// printing with cout << goes here.
// void Presolve::reportDebug(const string& message) const {

void print(const DevStats& stats) {
  std::cout << "dev-presolve-stats::" << std::endl;
  std::cout << "  n_loops = " << std::endl;
  std::cout << "    loop : rows, cols, nnz " << std::endl;
  for (const MainLoop l : stats.loops)
    std::cout << "    loop : " << l.rows << "," << l.cols << "," << l.nnz
              << "   " << std::endl;
  return;
}

void Presolve::reportDevMainLoop() {
  int rows = 0;
  int cols = 0;

  std::vector<int> nnz_rows(numRow, 0);
  std::vector<int> nnz_cols(numCol, 0);

  int total_rows = 0;
  int total_cols = 0;

  for (int i = 0; i < numRow; i++)
    if (flagRow.at(i)) {
      rows++;
      nnz_rows[i] += nzRow[i];
      total_rows += nzRow[i];
    }

  for (int j = 0; j < numCol; j++)
    if (flagCol.at(j)) {
      cols++;
      nnz_cols[j] += nzCol[j];
      total_cols += nzCol[j];
    }

  // Nonzeros.
  assert(total_cols == total_rows);

  dev_stats.n_loops++;
  dev_stats.loops.push_back(MainLoop{rows, cols, total_cols});
  return;
}

int Presolve::presolve(int print) {
  if (iPrint > 0) {
    cout << "Presolve started ..." << endl;
    cout << "Original problem ... N=" << numCol << "  M=" << numRow << endl;
  }

  if (iPrint < 0) {
    stringstream ss;
    ss << "dev-presolve: model:      rows, colx, nnz , " << modelName << ":  "
       << numRow << ",  " << numCol << ",  " << (int)Avalue.size() << endl;
    reportDev(ss.str());
  }

  initializeVectors();
  if (status) return status;

  int iter = 1;
  // print(0);

  timer.recordStart(FIXED_COL);
  assert((int)flagCol.size() == numCol);
  for (int j = 0; j < numCol; ++j)
    if (flagCol[j]) {
      removeIfFixed(j);
      if (status) return status;
    }
  timer.recordFinish(FIXED_COL);

  while (hasChange == 1) {
    hasChange = false;
    if (iPrint > 0) cout << "PR: main loop " << iter << ":" << endl;
    reportDevMainLoop();
    //***************** main loop ******************
    checkBoundsAreConsistent();
    if (status) return status;

    removeRowSingletons();
    if (status) return status;
    removeForcingConstraints(iter);
    if (status) return status;

    removeRowSingletons();
    if (status) return status;
    removeDoubletonEquations();
    if (status) return status;

    removeRowSingletons();
    if (status) return status;
    removeColumnSingletons();
    if (status) return status;

    removeDominatedColumns();
    if (status) return status;

    //***************** main loop ******************
    iter++;
  }

  timer.recordStart(RESIZE_MATRIX);
  checkForChanges(iter);
  timer.recordFinish(RESIZE_MATRIX);

  timer.updateInfo();

  return status;
}

HighsPresolveStatus Presolve::presolve() {
  timer.recordStart(TOTAL_PRESOLVE_TIME);
  HighsPresolveStatus presolve_status = HighsPresolveStatus::NotReduced;
  int result = presolve(0);
  switch (result) {
    case stat::Unbounded:
      presolve_status = HighsPresolveStatus::Unbounded;
      break;
    case stat::Infeasible:
      presolve_status = HighsPresolveStatus::Infeasible;
      break;
    case stat::Reduced:
      if (numCol > 0 || numRow > 0)
        presolve_status = HighsPresolveStatus::Reduced;
      else
        presolve_status = HighsPresolveStatus::ReducedToEmpty;
      break;
    case stat::Empty:
      presolve_status = HighsPresolveStatus::Empty;
      break;
    case stat::Optimal:
      // reduced problem solution indicated as optimal by
      // the solver.
      break;
  }
  timer.recordFinish(TOTAL_PRESOLVE_TIME);

  return presolve_status;
}

void Presolve::checkBoundsAreConsistent() {
  for (int col = 0; col < numCol; col++) {
    if (flagCol[col]) {
      if (colUpper[col] - colLower[col] < -tol) {
        status = Infeasible;
        return;
      }
    }
  }

  for (int row = 0; row < numRow; row++) {
    if (flagRow[row]) {
      if (rowUpper[row] - rowLower[row] < -tol) {
        status = Infeasible;
        return;
      }
    }
  }
}

/**
 * returns <x, y>
 * 		   <x, -1> if we need to skip row
 *
 * 		   row is of form akx_x + aky_y = b,
 */
pair<int, int> Presolve::getXYDoubletonEquations(const int row) {
  assert(row >= 0 && row < numRow);
  pair<int, int> colIndex;
  // row is of form akx_x + aky_y = b, where k=row and y is present in fewer
  // constraints

  int col1 = -1;
  int col2 = -1;
  int kk = ARstart[row];
  while (kk < ARstart[row + 1]) {
    // Was
    //
    // assert(ARindex[kk] < numCol && ARindex[kk] > 0);
    //
    // But a zero column index is surely OK, and a column index of
    // numCol is set in UpdateMatrixCoeffDoubletonEquationXnonZero
    // (line 549), with numCol a valid index of flagCol
    //
    assert(ARindex[kk] <= numCol && ARindex[kk] >= 0);
    if (flagCol[ARindex[kk]]) {
      if (col1 == -1)
        col1 = ARindex[kk];
      else if (col2 == -1)
        col2 = ARindex[kk];
      else {
        cout << "ERROR: doubleton eq row" << row
             << " has more than two variables. \n";
        col2 = -2;
        break;
      }
      ++kk;
    } else
      ++kk;
  }
  if (col2 == -1)
    cout << "ERROR: doubleton eq row" << row
         << " has less than two variables. \n";
  if (col2 < 0) {
    colIndex.second = -1;
    return colIndex;
  }

  int x, y;
  if (nzCol[col1] <= nzCol[col2]) {
    y = col1;
    x = col2;
  } else {
    x = col1;
    y = col2;
  }

  //	if (nzCol[y] == 1 && nzCol[x] == 1) { //two singletons case
  // handled elsewhere 		colIndex.second = -1; 		return colIndex;
  //	}

  colIndex.first = x;
  colIndex.second = y;
  return colIndex;
}

void Presolve::processRowDoubletonEquation(const int row, const int x,
                                           const int y, const double akx,
                                           const double aky, const double b) {
  assert(row < numRow && row >= 0);
  assert(x < numCol && x >= 0);
  assert(y < numCol && y >= 0);

  postValue.push(akx);
  postValue.push(aky);
  postValue.push(b);

  // modify bounds on variable x (j), variable y (col,k) is substituted out
  // double aik = Avalue[k];
  // double aij = Avalue[kk];
  pair<double, double> p = getNewBoundsDoubletonConstraint(row, y, x, aky, akx);
  double low = p.first;
  double upp = p.second;

  // add old bounds of x to checker and for postsolve
  if (iKKTcheck == 1) {
    vector<pair<int, double>> bndsL, bndsU, costS;
    bndsL.push_back(make_pair(x, colLower[x]));
    bndsU.push_back(make_pair(x, colUpper[x]));
    costS.push_back(make_pair(x, colCost[x]));
    chk.cLowers.push(bndsL);
    chk.cUppers.push(bndsU);
    chk.costs.push(costS);
  }

  vector<double> bnds({colLower[y], colUpper[y], colCost[y]});
  vector<double> bnds2({colLower[x], colUpper[x], colCost[x]});
  oldBounds.push(make_pair(y, bnds));
  oldBounds.push(make_pair(x, bnds2));

  if (low > colLower[x]) colLower[x] = low;
  if (upp < colUpper[x]) colUpper[x] = upp;

  // modify cost of xj
  colCost[x] = colCost[x] - colCost[y] * akx / aky;

  // for postsolve: need the new bounds too
  vector<double> bnds3({colLower[x], colUpper[x], colCost[x]});
  oldBounds.push(make_pair(x, bnds3));

  addChange(DOUBLETON_EQUATION, row, y);

  // remove y (col) and the row
  if (iPrint > 0)
    cout << "PR: Doubleton equation removed. Row " << row << ", column " << y
         << ", column left is " << x << "    nzy=" << nzCol[y] << endl;

  flagRow[row] = 0;
  nzCol[x]--;

  countRemovedRows(DOUBLETON_EQUATION);
  countRemovedCols(DOUBLETON_EQUATION);

  //----------------------------
  flagCol[y] = 0;
  if (!hasChange) hasChange = true;
}

void Presolve::removeDoubletonEquations() {
  timer.recordStart(DOUBLETON_EQUATION);
  // flagCol should have one more element at end which is zero
  // needed for AR matrix manipulation
  if ((int)flagCol.size() == numCol) flagCol.push_back(0);

  double b, akx, aky;
  int x, y;
  int iter = 0;

  for (int row = 0; row < numRow; row++)
    if (flagRow[row])
      if (nzRow[row] == 2 && rowLower[row] > -HIGHS_CONST_INF &&
          rowUpper[row] < HIGHS_CONST_INF &&
          fabs(rowLower[row] - rowUpper[row]) < tol) {
        // row is of form akx_x + aky_y = b, where k=row and y is present in
        // fewer constraints
        b = rowLower[row];
        pair<int, int> colIndex = getXYDoubletonEquations(row);
        x = colIndex.first;
        y = colIndex.second;

        // two singletons case handled elsewhere
        if (y < 0 || ((nzCol[y] == 1 && nzCol[x] == 1))) continue;

        akx = getaij(row, x);
        aky = getaij(row, y);
        processRowDoubletonEquation(row, x, y, akx, aky, b);

        for (int k = Astart[y]; k < Aend[y]; ++k)
          if (flagRow[Aindex[k]] && Aindex[k] != row) {
            int i = Aindex[k];
            double aiy = Avalue[k];

            // update row bounds
            if (iKKTcheck == 1) {
              vector<pair<int, double>> bndsL, bndsU;
              bndsL.push_back(make_pair(i, rowLower[i]));
              bndsU.push_back(make_pair(i, rowUpper[i]));
              chk.rLowers.push(bndsL);
              chk.rUppers.push(bndsU);
              addChange(DOUBLETON_EQUATION_ROW_BOUNDS_UPDATE, i, y);
            }

            if (rowLower[i] > -HIGHS_CONST_INF) rowLower[i] -= b * aiy / aky;
            if (rowUpper[i] < HIGHS_CONST_INF) rowUpper[i] -= b * aiy / aky;

            if (implRowValueLower[i] > -HIGHS_CONST_INF)
              implRowValueLower[i] -= b * aiy / aky;
            if (implRowValueUpper[i] < HIGHS_CONST_INF)
              implRowValueUpper[i] -= b * aiy / aky;

            // update matrix coefficients
            if (isZeroA(i, x))
              UpdateMatrixCoeffDoubletonEquationXzero(i, x, y, aiy, akx, aky);
            else
              UpdateMatrixCoeffDoubletonEquationXnonZero(i, x, y, aiy, akx,
                                                         aky);
          }
        if (Avalue.size() > 40000000) {
          trimA();
        }

        iter++;
      }
  timer.recordFinish(DOUBLETON_EQUATION);
}

void Presolve::UpdateMatrixCoeffDoubletonEquationXzero(const int i, const int x,
                                                       const int y,
                                                       const double aiy,
                                                       const double akx,
                                                       const double aky) {
  // case x is zero initially
  // row nonzero count doesn't change here
  // cout<<"case: x not present "<<i<<" "<<endl;

  // update AR
  int ind;
  for (ind = ARstart[i]; ind < ARstart[i + 1]; ++ind)
    if (ARindex[ind] == y) {
      break;
    }

  postValue.push(ARvalue[ind]);
  postValue.push(y);
  addChange(DOUBLETON_EQUATION_X_ZERO_INITIALLY, i, x);

  ARindex[ind] = x;
  ARvalue[ind] = -aiy * akx / aky;

  // just row rep in checker
  if (iKKTcheck == 1) {
    chk.ARvalue[ind] = ARvalue[ind];
    chk.ARindex[ind] = ARindex[ind];
  }

  // update A: append X column to end of array
  int st = Avalue.size();
  for (int ind = Astart[x]; ind < Aend[x]; ++ind) {
    Avalue.push_back(Avalue[ind]);
    Aindex.push_back(Aindex[ind]);
  }
  Avalue.push_back(-aiy * akx / aky);
  Aindex.push_back(i);
  Astart[x] = st;
  Aend[x] = Avalue.size();

  nzCol[x]++;
  // nzRow does not change here.
  if (nzCol[x] == 2) singCol.remove(x);
}

void Presolve::UpdateMatrixCoeffDoubletonEquationXnonZero(
    const int i, const int x, const int y, const double aiy, const double akx,
    const double aky) {
  int ind;

  // update nonzeros: for removal of
  nzRow[i]--;
  if (nzRow[i] == 1) singRow.push_back(i);

  if (nzRow[i] == 0) {
    singRow.remove(i);
    removeEmptyRow(i);
    countRemovedRows(DOUBLETON_EQUATION);
  }

  double xNew;
  for (ind = ARstart[i]; ind < ARstart[i + 1]; ++ind)
    if (ARindex[ind] == x) break;

  xNew = ARvalue[ind] - (aiy * akx) / aky;
  if (fabs(xNew) > tol) {
    // case new x != 0
    // cout<<"case: x still there row "<<i<<" "<<endl;

    postValue.push(ARvalue[ind]);
    addChange(DOUBLETON_EQUATION_NEW_X_NONZERO, i, x);
    ARvalue[ind] = xNew;

    if (iKKTcheck == 1) chk.ARvalue[ind] = xNew;

    // update A:
    for (ind = Astart[x]; ind < Aend[x]; ++ind)
      if (Aindex[ind] == i) {
        break;
      }
    Avalue[ind] = xNew;
  } else if (xNew < tol) {
    // case new x == 0
    // cout<<"case: x also disappears from row "<<i<<" "<<endl;
    // update nz row
    nzRow[i]--;
    // update singleton row list
    if (nzRow[i] == 1) singRow.push_back(i);

    if (nzRow[i] == 0) {
      singRow.remove(i);
      removeEmptyRow(i);
      countRemovedRows(DOUBLETON_EQUATION);
    }

    if (nzRow[i] > 0) {
      // AR update
      // set ARindex of element for x to numCol
      // flagCol[numCol] = false
      // mind when resizing: should be OK
      postValue.push(ARvalue[ind]);

      ARindex[ind] = numCol;
      if (iKKTcheck == 1) {
        chk.ARindex[ind] = ARindex[ind];
        chk.ARvalue[ind] = ARvalue[ind];
      }

      addChange(DOUBLETON_EQUATION_NEW_X_ZERO_AR_UPDATE, i, x);
    }

    if (nzCol[x] > 0) {
      // A update for case when x is zero: move x entry to end and set
      // Aend to be Aend - 1;
      int indi;
      for (indi = Astart[x]; indi < Aend[x]; ++indi)
        if (Aindex[indi] == i) break;

      postValue.push(Avalue[indi]);

      // if indi is not Aend-1 swap elements indi and Aend-1
      if (indi != Aend[x] - 1) {
        double tmp = Avalue[Aend[x] - 1];
        int tmpi = Aindex[Aend[x] - 1];
        Avalue[Aend[x] - 1] = Avalue[indi];
        Aindex[Aend[x] - 1] = Aindex[indi];
        Avalue[indi] = tmp;
        Aindex[indi] = tmpi;
      }
      Aend[x]--;
      addChange(DOUBLETON_EQUATION_NEW_X_ZERO_A_UPDATE, i, x);
    }

    // update nz col
    nzCol[x]--;
    // update singleton col list
    if (nzCol[x] == 1) singCol.push_back(x);
    if (nzCol[x] == 0) {
      removeEmptyColumn(x);
    }
  }
  if (y) {
  }  // surpress warning.
}

void Presolve::trimA() {
  int cntEl = 0;
  for (int j = 0; j < numCol; ++j)
    if (flagCol[j]) cntEl += nzCol[j];

  vector<pair<int, size_t>> vp;
  vp.reserve(numCol);

  for (int i = 0; i != numCol; ++i) {
    vp.push_back(make_pair(Astart[i], i));
  }

  // Sorting will put lower values ahead of larger ones,
  // resolving ties using the original index
  sort(vp.begin(), vp.end());

  vector<int> Aendtmp;
  Aendtmp = Aend;

  int iPut = 0;
  for (size_t i = 0; i != vp.size(); ++i) {
    int col = vp[i].second;
    if (flagCol[col]) {
      int k = vp[i].first;
      Astart[col] = iPut;
      while (k < Aendtmp[col]) {
        if (flagRow[Aindex[k]]) {
          Avalue[iPut] = Avalue[k];
          Aindex[iPut] = Aindex[k];
          iPut++;
        }
        k++;
      }
      Aend[col] = iPut;
    }
  }
  Avalue.resize(iPut);
  Aindex.resize(iPut);
}

void Presolve::resizeProblem() {
  int i, j, k;

  int nz = 0;
  int nR = 0;
  int nC = 0;

  // arrays to keep track of indices
  rIndex.assign(numRow, -1);
  cIndex.assign(numCol, -1);

  for (i = 0; i < numRow; ++i)
    if (flagRow[i]) {
      nz += nzRow[i];
      rIndex[i] = nR;
      nR++;
    }

  for (i = 0; i < numCol; ++i)
    if (flagCol[i]) {
      cIndex[i] = nC;
      nC++;
    }

  // counts
  numRowOriginal = numRow;
  numColOriginal = numCol;
  numRow = nR;
  numCol = nC;
  numTot = nR + nC;

  if (iPrint < 0) {
    stringstream ss;
    ss << ",  Reduced : " << numRow << ",  " << numCol << ",  ";
    reportDev(ss.str());
  }

  if (nR + nC == 0) {
    status = Empty;
    return;
  }

  // matrix
  vector<int> iwork(numCol, 0);
  Astart.assign(numCol + 1, 0);
  Aend.assign(numCol + 1, 0);
  Aindex.resize(nz);
  Avalue.resize(nz);

  for (i = 0; i < numRowOriginal; ++i)
    if (flagRow[i])
      for (int k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        j = ARindex[k];
        if (flagCol[j]) iwork[cIndex[j]]++;
      }
  for (i = 1; i <= numCol; ++i) Astart[i] = Astart[i - 1] + iwork[i - 1];
  for (i = 0; i < numCol; ++i) iwork[i] = Aend[i] = Astart[i];
  for (i = 0; i < numRowOriginal; ++i) {
    if (flagRow[i]) {
      int iRow = rIndex[i];
      for (k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        j = ARindex[k];
        if (flagCol[j]) {
          int iCol = cIndex[j];
          int iPut = iwork[iCol]++;
          Aindex[iPut] = iRow;
          Avalue[iPut] = ARvalue[k];
        }
      }
    }
  }

  if (iPrint < 0) {
    stringstream ss;
    ss << Avalue.size() << ", ";
    reportDev(ss.str());
  }

  // For KKT checker: pass vectors before you trim them
  if (iKKTcheck == 1) {
    chk.setFlags(flagRow, flagCol);
    chk.setBoundsCostRHS(colUpper, colLower, colCost, rowLower, rowUpper);
  }

  // also call before trimming
  resizeImpliedBounds();

  // cost, bounds
  colCostAtEl = colCost;
  vector<double> tempCost = colCost;
  vector<double> temp = colLower;
  vector<double> teup = colUpper;

  colCost.resize(numCol);
  colLower.resize(numCol);
  colUpper.resize(numCol);

  k = 0;
  for (i = 0; i < numColOriginal; ++i)
    if (flagCol[i]) {
      colCost[k] = tempCost[i];
      colLower[k] = temp[i];
      colUpper[k] = teup[i];
      k++;
    }

  // RHS and bounds
  rowLowerAtEl = rowLower;
  rowUpperAtEl = rowUpper;
  temp = rowLower;
  teup = rowUpper;
  rowLower.resize(numRow);
  rowUpper.resize(numRow);
  k = 0;
  for (i = 0; i < numRowOriginal; ++i)
    if (flagRow[i]) {
      rowLower[k] = temp[i];
      rowUpper[k] = teup[i];
      k++;
    }

  if (chk.print == 3) {
    ofstream myfile;
    myfile.open("../experiments/out", ios::app);
    myfile << " eliminated rows " << (numRowOriginal - numRow) << " cols "
           << (numColOriginal - numCol);
    myfile.close();

    myfile.open("../experiments/t3", ios::app);
    myfile << (numRowOriginal) << "  &  " << (numColOriginal) << "  & ";
    myfile << (numRowOriginal - numRow) << "  &  " << (numColOriginal - numCol)
           << "  & " << endl;

    myfile.close();
  }
}

void Presolve::initializeVectors() {
  // copy original bounds
  colCostOriginal = colCost;
  rowUpperOriginal = rowUpper;
  rowLowerOriginal = rowLower;
  colUpperOriginal = colUpper;
  colLowerOriginal = colLower;

  makeARCopy();

  valueRowDual.resize(numRow);
  valuePrimal.resize(numCol);
  valueColDual.resize(numCol);

  flagCol.assign(numCol, 1);
  flagRow.assign(numRow, 1);

  if (iKKTcheck) setKKTcheckerData();

  nzCol.assign(numCol, 0);
  nzRow.assign(numRow, 0);

  for (int i = 0; i < numRow; ++i) {
    nzRow[i] = ARstart[i + 1] - ARstart[i];
    if (nzRow[i] == 1) singRow.push_back(i);
    if (nzRow[i] == 0) {
      timer.recordStart(EMPTY_ROW);
      removeEmptyRow(i);
      countRemovedRows(EMPTY_ROW);
      timer.recordFinish(EMPTY_ROW);
    }
  }

  Aend.resize(numCol + 1);
  for (int i = 0; i < numCol; ++i) {
    Aend[i] = Astart[i + 1];
    nzCol[i] = Aend[i] - Astart[i];
    if (nzCol[i] == 1) singCol.push_back(i);
  }
  objShift = 0;

  implColUpper = colUpper;  // working copies of primal variable bounds
  implColLower = colLower;
  implColLowerRowIndex.assign(numCol, -1);
  implColUpperRowIndex.assign(numCol, -1);

  implRowDualLowerSingColRowIndex.assign(numRow, -1);
  implRowDualUpperSingColRowIndex.assign(numRow, -1);
  implRowDualLower.assign(numRow, -HIGHS_CONST_INF);
  implRowDualUpper.assign(numRow, HIGHS_CONST_INF);

  implColDualLower.assign(numCol, -HIGHS_CONST_INF);
  implColDualUpper.assign(numCol, HIGHS_CONST_INF);
  implRowValueLower = rowLower;
  implRowValueUpper = rowUpper;

  for (int i = 0; i < numRow; ++i) {
    if (rowLower[i] <= -HIGHS_CONST_INF) implRowDualUpper[i] = 0;
    if (rowUpper[i] >= HIGHS_CONST_INF) implRowDualLower[i] = 0;
  }

  for (int i = 0; i < numCol; ++i) {
    if (colLower[i] <= -HIGHS_CONST_INF) implColDualUpper[i] = 0;
    if (colUpper[i] >= HIGHS_CONST_INF) implColDualLower[i] = 0;
  }

  colCostAtEl = colCost;
  rowLowerAtEl = rowLower;
  rowUpperAtEl = rowUpper;
}

void Presolve::removeIfFixed(int j) {
  if (colLower[j] == colUpper[j]) {
    setPrimalValue(j, colUpper[j]);
    addChange(FIXED_COL, 0, j);
    if (iPrint > 0)
      cout << "PR: Fixed variable " << j << " = " << colUpper[j]
           << ". Column eliminated." << endl;

    countRemovedCols(FIXED_COL);

    for (int k = Astart[j]; k < Aend[j]; ++k) {
      if (flagRow[Aindex[k]]) {
        int i = Aindex[k];

        if (nzRow[i] == 0) {
          removeEmptyRow(i);
          countRemovedRows(FIXED_COL);
        }
      }
    }
  }
}

void Presolve::removeEmptyRow(int i) {
  if (rowLower[i] <= tol && rowUpper[i] >= -tol) {
    if (iPrint > 0) cout << "PR: Empty row " << i << " removed. " << endl;
    flagRow[i] = 0;
    valueRowDual[i] = 0;
    addChange(EMPTY_ROW, i, 0);
  } else {
    if (iPrint > 0) cout << "PR: Problem infeasible." << endl;
    status = Infeasible;
    return;
  }
}

void Presolve::removeEmptyColumn(int j) {
  flagCol[j] = 0;
  singCol.remove(j);
  double value;
  if ((colCost[j] < 0 && colUpper[j] >= HIGHS_CONST_INF) ||
      (colCost[j] > 0 && colLower[j] <= -HIGHS_CONST_INF)) {
    if (iPrint > 0) cout << "PR: Problem unbounded." << endl;
    status = Unbounded;
    return;
  }

  if (colCost[j] > 0)
    value = colLower[j];
  else if (colCost[j] < 0)
    value = colUpper[j];
  else if (colUpper[j] >= 0 && colLower[j] <= 0)
    value = 0;
  else if (colUpper[j] < 0)
    value = colUpper[j];
  else
    value = colLower[j];

  setPrimalValue(j, value);
  valueColDual[j] = colCost[j];

  addChange(EMPTY_COL, 0, j);

  if (iPrint > 0)
    cout << "PR: Column: " << j
         << " eliminated: all nonzero rows have been removed. Cost = "
         << colCost[j] << ", value = " << value << endl;

  countRemovedCols(EMPTY_COL);
}

void Presolve::rowDualBoundsDominatedColumns() {
  int col, i, k;

  // for each row calc yihat and yibar and store in implRowDualLower and
  // implRowDualUpper
  for (list<int>::iterator it = singCol.begin(); it != singCol.end(); ++it)
    if (flagCol[*it]) {
      col = *it;
      k = getSingColElementIndexInA(col);
      i = Aindex[k];

      if (!flagRow[i]) {
        cout << "ERROR: column singleton " << col << " is in row " << i
             << " which is already mapped off\n";
        exit(-1);
      }

      if (colLower[col] <= -HIGHS_CONST_INF ||
          colUpper[col] >= HIGHS_CONST_INF) {
        if (colLower[col] > -HIGHS_CONST_INF &&
            colUpper[col] >= HIGHS_CONST_INF) {
          if (Avalue[k] > 0)
            if ((colCost[col] / Avalue[k]) < implRowDualUpper[i])
              implRowDualUpper[i] = colCost[col] / Avalue[k];
          if (Avalue[k] < 0)
            if ((colCost[col] / Avalue[k]) > implRowDualLower[i])
              implRowDualLower[i] = colCost[col] / Avalue[k];
        } else if (colLower[col] <= -HIGHS_CONST_INF &&
                   colUpper[col] < HIGHS_CONST_INF) {
          if (Avalue[k] > 0)
            if ((colCost[col] / Avalue[k]) > implRowDualLower[i])
              implRowDualUpper[i] = -colCost[col] / Avalue[k];
          if (Avalue[k] < 0)
            if ((colCost[col] / Avalue[k]) < implRowDualUpper[i])
              implRowDualUpper[i] = colCost[col] / Avalue[k];
        } else if (colLower[col] <= -HIGHS_CONST_INF &&
                   colUpper[col] >= HIGHS_CONST_INF) {
          // all should be removed earlier but use them
          if ((colCost[col] / Avalue[k]) > implRowDualLower[i])
            implRowDualLower[i] = colCost[col] / Avalue[k];
          if ((colCost[col] / Avalue[k]) < implRowDualUpper[i])
            implRowDualUpper[i] = colCost[col] / Avalue[k];
        }

        if (implRowDualLower[i] > implRowDualUpper[i]) {
          cout << "Error: inconstistent bounds for Lagrange multiplier for row "
               << i << " detected after column singleton " << col
               << ". In presolve::dominatedColumns" << endl;
          exit(0);
        }
      }
    }
}

pair<double, double> Presolve::getImpliedColumnBounds(int j) {
  pair<double, double> out;
  double e = 0;
  double d = 0;

  int i;
  for (int k = Astart[j]; k < Aend[j]; ++k) {
    i = Aindex[k];
    if (flagRow[i]) {
      if (Avalue[k] < 0) {
        if (implRowDualUpper[i] < HIGHS_CONST_INF)
          e += Avalue[k] * implRowDualUpper[i];
        else {
          e = -HIGHS_CONST_INF;
          break;
        }
      } else {
        if (implRowDualLower[i] > -HIGHS_CONST_INF)
          e += Avalue[k] * implRowDualLower[i];
        else {
          e = -HIGHS_CONST_INF;
          break;
        }
      }
    }
  }

  for (int k = Astart[j]; k < Aend[j]; ++k) {
    i = Aindex[k];
    if (flagRow[i]) {
      if (Avalue[k] < 0) {
        if (implRowDualLower[i] > -HIGHS_CONST_INF)
          d += Avalue[k] * implRowDualLower[i];
        else {
          d = HIGHS_CONST_INF;
          break;
        }
      } else {
        if (implRowDualUpper[i] < HIGHS_CONST_INF)
          d += Avalue[k] * implRowDualUpper[i];
        else {
          d = HIGHS_CONST_INF;
          break;
        }
      }
    }
  }

  if (e > d) {
    cout << "Error: inconstistent bounds for Lagrange multipliers for column "
         << j << ": e>d. In presolve::dominatedColumns" << endl;
    exit(-1);
  }
  out.first = d;
  out.second = e;
  return out;
}

void Presolve::removeDominatedColumns() {
  // for each column j calculate e and d and check:
  double e, d;
  pair<double, double> p;
  for (int j = 0; j < numCol; ++j)
    if (flagCol[j]) {
      timer.recordStart(DOMINATED_COLS);

      p = getImpliedColumnBounds(j);
      d = p.first;
      e = p.second;

      // check if it is dominated
      if (colCost[j] - d > tol) {
        if (colLower[j] <= -HIGHS_CONST_INF) {
          if (iPrint > 0) cout << "PR: Problem unbounded." << endl;
          status = Unbounded;
          return;
        }
        setPrimalValue(j, colLower[j]);
        addChange(DOMINATED_COLS, 0, j);
        if (iPrint > 0)
          cout << "PR: Dominated column " << j
               << " removed. Value := " << valuePrimal[j] << endl;
        countRemovedCols(DOMINATED_COLS);
      } else if (colCost[j] - e < -tol) {
        if (colUpper[j] >= HIGHS_CONST_INF) {
          if (iPrint > 0) cout << "PR: Problem unbounded." << endl;
          status = Unbounded;
          return;
        }
        setPrimalValue(j, colUpper[j]);
        addChange(DOMINATED_COLS, 0, j);
        if (iPrint > 0)
          cout << "PR: Dominated column " << j
               << " removed. Value := " << valuePrimal[j] << endl;
        countRemovedCols(DOMINATED_COLS);
      } else {
        // update implied bounds
        if (implColDualLower[j] < (colCost[j] - d))
          implColDualLower[j] = colCost[j] - d;
        if (implColDualUpper[j] > (colCost[j] - e))
          implColDualUpper[j] = colCost[j] - e;
        if (implColDualLower[j] > implColDualUpper[j]) cout << "INCONSISTENT\n";

        timer.recordFinish(DOMINATED_COLS);

        removeIfWeaklyDominated(j, d, e);
        continue;
      }
      timer.recordFinish(DOMINATED_COLS);
    }
}

void Presolve::removeIfWeaklyDominated(const int j, const double d,
                                       const double e) {
  timer.recordStart(WEAKLY_DOMINATED_COLS);

  int i;
  // check if it is weakly dominated: Excluding singletons!
  if (nzCol[j] > 1) {
    if (d < HIGHS_CONST_INF && fabs(colCost[j] - d) < tol &&
        colLower[j] > -HIGHS_CONST_INF) {
      setPrimalValue(j, colLower[j]);
      addChange(WEAKLY_DOMINATED_COLS, 0, j);
      if (iPrint > 0)
        cout << "PR: Weakly Dominated column " << j
             << " removed. Value := " << valuePrimal[j] << endl;

      countRemovedCols(WEAKLY_DOMINATED_COLS);
    } else if (e > -HIGHS_CONST_INF && fabs(colCost[j] - e) < tol &&
               colUpper[j] < HIGHS_CONST_INF) {
      setPrimalValue(j, colUpper[j]);
      addChange(WEAKLY_DOMINATED_COLS, 0, j);
      if (iPrint > 0)
        cout << "PR: Weakly Dominated column " << j
             << " removed. Value := " << valuePrimal[j] << endl;

      countRemovedCols(WEAKLY_DOMINATED_COLS);
    } else {
      double bnd;

      // calculate new bounds
      if (colLower[j] > -HIGHS_CONST_INF || colUpper[j] >= HIGHS_CONST_INF)
        for (int kk = Astart[j]; kk < Aend[j]; ++kk)
          if (flagRow[Aindex[kk]] && d < HIGHS_CONST_INF) {
            i = Aindex[kk];
            if (Avalue[kk] > 0 && implRowDualLower[i] > -HIGHS_CONST_INF) {
              bnd = -(colCost[j] + d) / Avalue[kk] + implRowDualLower[i];
              if (bnd < implRowDualUpper[i] && !(bnd < implRowDualLower[i]))
                implRowDualUpper[i] = bnd;
            } else if (Avalue[kk] < 0 &&
                       implRowDualUpper[i] < HIGHS_CONST_INF) {
              bnd = -(colCost[j] + d) / Avalue[kk] + implRowDualUpper[i];
              if (bnd > implRowDualLower[i] && !(bnd > implRowDualUpper[i]))
                implRowDualLower[i] = bnd;
            }
          }

      if (colLower[j] <= -HIGHS_CONST_INF || colUpper[j] < HIGHS_CONST_INF)
        for (int kk = Astart[j]; kk < Aend[j]; ++kk)
          if (flagRow[Aindex[kk]] && e > -HIGHS_CONST_INF) {
            i = Aindex[kk];
            if (Avalue[kk] > 0 && implRowDualUpper[i] < HIGHS_CONST_INF) {
              bnd = -(colCost[j] + e) / Avalue[kk] + implRowDualUpper[i];
              if (bnd > implRowDualLower[i] && !(bnd > implRowDualUpper[i]))
                implRowDualLower[i] = bnd;
            } else if (Avalue[kk] < 0 &&
                       implRowDualLower[i] > -HIGHS_CONST_INF) {
              bnd = -(colCost[j] + e) / Avalue[kk] + implRowDualLower[i];
              if (bnd < implRowDualUpper[i] && !(bnd < implRowDualLower[i]))
                implRowDualUpper[i] = bnd;
            }
          }
    }
  }
  timer.recordFinish(WEAKLY_DOMINATED_COLS);
}

void Presolve::setProblemStatus(const int s) {
  if (s == Infeasible)
    cout << "NOT-OPT status = 1, returned from solver after presolve: Problem "
            "infeasible.\n";
  else if (s == Unbounded)
    cout << "NOT-OPT status = 2, returned from solver after presolve: Problem "
            "unbounded.\n";
  else if (s == 0) {
    status = Optimal;
    return;
  } else
    cout << "unknown problem status returned from solver after presolve: " << s
         << endl;
  status = s;
}

void Presolve::setKKTcheckerData() {
  // after initializing equations.
  chk.setMatrixAR(numCol, numRow, ARstart, ARindex, ARvalue);
  chk.setFlags(flagRow, flagCol);
  chk.setBoundsCostRHS(colUpper, colLower, colCost, rowLower, rowUpper);
}

pair<double, double> Presolve::getNewBoundsDoubletonConstraint(int row, int col,
                                                               int j,
                                                               double aik,
                                                               double aij) {
  int i = row;

  double upp = HIGHS_CONST_INF;
  double low = -HIGHS_CONST_INF;

  if (aij > 0 && aik > 0) {
    if (colLower[col] > -HIGHS_CONST_INF)
      upp = (rowUpper[i] - aik * colLower[col]) / aij;
    if (colUpper[col] < HIGHS_CONST_INF)
      low = (rowLower[i] - aik * colUpper[col]) / aij;
  } else if (aij > 0 && aik < 0) {
    if (colLower[col] > -HIGHS_CONST_INF)
      low = (rowLower[i] - aik * colLower[col]) / aij;
    if (colUpper[col] < HIGHS_CONST_INF)
      upp = (rowUpper[i] - aik * colUpper[col]) / aij;
  } else if (aij < 0 && aik > 0) {
    if (colLower[col] > -HIGHS_CONST_INF)
      low = (rowUpper[i] - aik * colLower[col]) / aij;
    if (colUpper[col] < HIGHS_CONST_INF)
      upp = (rowLower[i] - aik * colUpper[col]) / aij;
  } else {
    if (colLower[col] > -HIGHS_CONST_INF)
      upp = (rowLower[i] - aik * colLower[col]) / aij;
    if (colUpper[col] < HIGHS_CONST_INF)
      low = (rowUpper[i] - aik * colUpper[col]) / aij;
  }

  if (j) {
  }  // surpress warning.

  return make_pair(low, upp);
}

void Presolve::removeFreeColumnSingleton(const int col, const int row,
                                         const int k) {
  if (iPrint > 0)
    cout << "PR: Free column singleton " << col << " removed. Row " << row
         << " removed." << endl;

  // modify costs
  vector<pair<int, double>> newCosts;
  int j;
  for (int kk = ARstart[row]; kk < ARstart[row + 1]; ++kk) {
    j = ARindex[kk];
    if (flagCol[j] && j != col) {
      newCosts.push_back(make_pair(j, colCost[j]));
      colCost[j] = colCost[j] - colCost[col] * ARvalue[kk] / Avalue[k];
    }
  }
  if (iKKTcheck == 1) chk.costs.push(newCosts);

  flagCol[col] = 0;
  postValue.push(colCost[col]);
  fillStackRowBounds(row);

  valueColDual[col] = 0;
  valueRowDual[row] = -colCost[col] / Avalue[k];

  addChange(FREE_SING_COL, row, col);
  removeRow(row);

  countRemovedCols(FREE_SING_COL);
  countRemovedRows(FREE_SING_COL);
}

bool Presolve::removeColumnSingletonInDoubletonInequality(const int col,
                                                          const int i,
                                                          const int k) {
  // second column index j
  // second column row array index kk
  int j = -1;

  // count
  int kk = ARstart[i];
  while (kk < ARstart[i + 1]) {
    j = ARindex[kk];
    if (flagCol[j] && j != col)
      break;
    else
      ++kk;
  }
  if (kk == ARstart[i + 1])
    cout << "ERROR: nzRow[" << i << "]=2, but no second variable in row. \n";

  // only inequality case and case two singletons here,
  // others handled in doubleton equation
  if ((fabs(rowLower[i] - rowUpper[i]) < tol) && (nzCol[j] > 1)) return false;

  // additional check if it is indeed implied free
  // needed since we handle inequalities and it may not be true
  // low and upp to be tighter than original bounds for variable col
  // so it is indeed implied free and we can remove it
  pair<double, double> p =
      getNewBoundsDoubletonConstraint(i, j, col, ARvalue[kk], Avalue[k]);
  if (!(colLower[col] <= p.first && colUpper[col] >= p.second)) {
    return false;
  }

  postValue.push(ARvalue[kk]);
  postValue.push(Avalue[k]);

  // modify bounds on variable j, variable col (k) is substituted out
  // double aik = Avalue[k];
  // double aij = Avalue[kk];
  p = getNewBoundsDoubletonConstraint(i, col, j, Avalue[k], ARvalue[kk]);
  double low = p.first;
  double upp = p.second;

  // add old bounds of xj to checker and for postsolve
  if (iKKTcheck == 1) {
    vector<pair<int, double>> bndsL, bndsU, costS;
    bndsL.push_back(make_pair(j, colLower[j]));
    bndsU.push_back(make_pair(j, colUpper[j]));
    costS.push_back(make_pair(j, colCost[j]));
    chk.cLowers.push(bndsL);
    chk.cUppers.push(bndsU);
    chk.costs.push(costS);
  }

  vector<double> bndsCol({colLower[col], colUpper[col], colCost[col]});
  vector<double> bndsJ({colLower[j], colUpper[j], colCost[j]});
  oldBounds.push(make_pair(col, bndsCol));
  oldBounds.push(make_pair(j, bndsJ));

  // modify bounds of xj
  if (low > colLower[j]) colLower[j] = low;
  if (upp < colUpper[j]) colUpper[j] = upp;

  // modify cost of xj
  colCost[j] = colCost[j] - colCost[col] * ARvalue[kk] / Avalue[k];

  // for postsolve: need the new bounds too
  // oldBounds.push_back(colLower[j]); oldBounds.push_back(colUpper[j]);
  bndsJ[0] = (colLower[j]);
  bndsJ[1] = (colUpper[j]);
  bndsJ[2] = (colCost[j]);
  oldBounds.push(make_pair(j, bndsJ));

  // remove col as free column singleton
  if (iPrint > 0)
    cout << "PR: Column singleton " << col
         << " in a doubleton inequality constraint removed. Row " << i
         << " removed. variable left is " << j << endl;

  flagCol[col] = 0;
  fillStackRowBounds(i);
  countRemovedCols(SING_COL_DOUBLETON_INEQ);
  countRemovedRows(SING_COL_DOUBLETON_INEQ);

  valueColDual[col] = 0;
  valueRowDual[i] =
      -colCost[col] / Avalue[k];  // may be changed later, depending on bounds.
  addChange(SING_COL_DOUBLETON_INEQ, i, col);

  // if not special case two column singletons
  if (nzCol[j] > 1)
    removeRow(i);
  else if (nzCol[j] == 1)
    removeSecondColumnSingletonInDoubletonRow(j, i);

  return true;
}

void Presolve::removeSecondColumnSingletonInDoubletonRow(const int j,
                                                         const int i) {
  // case two singleton columns
  // when we get here bounds on xj are updated so we can choose low/upper one
  // depending on the cost of xj
  flagRow[i] = 0;
  double value;
  if (colCost[j] > 0) {
    if (colLower[j] <= -HIGHS_CONST_INF) {
      if (iPrint > 0) cout << "PR: Problem unbounded." << endl;
      status = Unbounded;
      return;
    }
    value = colLower[j];
  } else if (colCost[j] < 0) {
    if (colUpper[j] >= HIGHS_CONST_INF) {
      if (iPrint > 0) cout << "PR: Problem unbounded." << endl;
      status = Unbounded;
      return;
    }
    value = colUpper[j];
  } else {  //(colCost[j] == 0)
    if (colUpper[j] >= 0 && colLower[j] <= 0)
      value = 0;
    else if (fabs(colUpper[j]) < fabs(colLower[j]))
      value = colUpper[j];
    else
      value = colLower[j];
  }
  setPrimalValue(j, value);
  addChange(SING_COL_DOUBLETON_INEQ_SECOND_SING_COL, 0, j);
  if (iPrint > 0)
    cout << "PR: Second singleton column " << j << " in doubleton row " << i
         << " removed.\n";
  countRemovedCols(SING_COL_DOUBLETON_INEQ);
  singCol.remove(j);
}

void Presolve::removeColumnSingletons() {
  int i, k, col;
  list<int>::iterator it = singCol.begin();

  while (it != singCol.end()) {
    if (flagCol[*it]) {
      col = *it;
      k = getSingColElementIndexInA(col);
      i = Aindex[k];

      // free
      if (colLower[col] <= -HIGHS_CONST_INF &&
          colUpper[col] >= HIGHS_CONST_INF) {
        timer.recordStart(FREE_SING_COL);
        removeFreeColumnSingleton(col, i, k);
        it = singCol.erase(it);
        timer.recordFinish(FREE_SING_COL);
        continue;
      }
      // singleton column in a doubleton inequality
      // case two column singletons
      else if (nzRow[i] == 2) {
        timer.recordStart(SING_COL_DOUBLETON_INEQ);
        bool result = removeColumnSingletonInDoubletonInequality(col, i, k);
        timer.recordFinish(SING_COL_DOUBLETON_INEQ);
        if (result) {
          it = singCol.erase(it);
          continue;
        }
      }
      // implied free
      else {
        timer.recordStart(IMPLIED_FREE_SING_COL);
        bool result = removeIfImpliedFree(col, i, k);
        timer.recordFinish(IMPLIED_FREE_SING_COL);
        if (result) {
          it = singCol.erase(it);
          continue;
        }
      }
      it++;
    } else
      it = singCol.erase(it);
  }
}

pair<double, double> Presolve::getBoundsImpliedFree(double lowInit,
                                                    double uppInit,
                                                    const int col, const int i,
                                                    const int k) {
  double low = lowInit;
  double upp = uppInit;

  // use implied bounds with original bounds
  int j;
  double l, u;
  // if at any stage low becomes  or upp becomes inf break loop
  // can't use bounds for variables generated by the same row.
  // low
  for (int kk = ARstart[i]; kk < ARstart[i + 1]; ++kk) {
    j = ARindex[kk];
    if (flagCol[j] && j != col) {
      // check if new bounds are precisely implied bounds from same row
      if (i != implColLowerRowIndex[j])
        l = max(colLower[j], implColLower[j]);
      else
        l = colLower[j];
      if (i != implColUpperRowIndex[j])
        u = min(colUpper[j], implColUpper[j]);
      else
        u = colUpper[j];

      if ((Avalue[k] < 0 && ARvalue[kk] > 0) ||
          (Avalue[k] > 0 && ARvalue[kk] < 0))
        if (l <= -HIGHS_CONST_INF) {
          low = -HIGHS_CONST_INF;
          break;
        } else
          low -= ARvalue[kk] * l;
      else if (u >= HIGHS_CONST_INF) {
        low = -HIGHS_CONST_INF;
        break;
      } else
        low -= ARvalue[kk] * u;
    }
  }
  // upp
  for (int kk = ARstart[i]; kk < ARstart[i + 1]; ++kk) {
    j = ARindex[kk];
    if (flagCol[j] && j != col) {
      // check if new bounds are precisely implied bounds from same row
      if (i != implColLowerRowIndex[j])
        l = max(colLower[j], implColLower[j]);
      else
        l = colLower[j];
      if (i != implColUpperRowIndex[j])
        u = min(colUpper[j], implColUpper[j]);
      else
        u = colUpper[j];
      // if at any stage low becomes  or upp becomes inf it's not implied free
      // low::
      if ((Avalue[k] < 0 && ARvalue[kk] > 0) ||
          (Avalue[k] > 0 && ARvalue[kk] < 0))
        if (u >= HIGHS_CONST_INF) {
          upp = HIGHS_CONST_INF;
          break;
        } else
          upp -= ARvalue[kk] * u;
      else if (l <= -HIGHS_CONST_INF) {
        upp = HIGHS_CONST_INF;
        break;
      } else
        upp -= ARvalue[kk] * l;
    }
  }
  return make_pair(low, upp);
}

void Presolve::removeImpliedFreeColumn(const int col, const int i,
                                       const int k) {
  if (iPrint > 0)
    cout << "PR: Implied free column singleton " << col << " removed.  Row "
         << i << " removed." << endl;

  countRemovedCols(IMPLIED_FREE_SING_COL);
  countRemovedRows(IMPLIED_FREE_SING_COL);

  // modify costs
  int j;
  vector<pair<int, double>> newCosts;
  for (int kk = ARstart[i]; kk < ARstart[i + 1]; ++kk) {
    j = ARindex[kk];
    if (flagCol[j] && j != col) {
      newCosts.push_back(make_pair(j, colCost[j]));
      colCost[j] = colCost[j] - colCost[col] * ARvalue[kk] / Avalue[k];
    }
  }
  if (iKKTcheck == 1) chk.costs.push(newCosts);

  flagCol[col] = 0;
  postValue.push(colCost[col]);
  fillStackRowBounds(i);

  valueColDual[col] = 0;
  valueRowDual[i] = -colCost[col] / Avalue[k];
  addChange(IMPLIED_FREE_SING_COL, i, col);
  removeRow(i);
}

bool Presolve::removeIfImpliedFree(int col, int i, int k) {
  // first find which bound is active for row i
  // A'y + c = z so yi = -ci/aij
  double aij = getaij(i, col);
  if (aij != Avalue[k]) cout << "ERROR during implied free";
  double yi = -colCost[col] / aij;
  double low, upp;

  if (yi > 0) {
    if (rowUpper[i] >= HIGHS_CONST_INF) return false;
    low = rowUpper[i];
    upp = rowUpper[i];
  } else if (yi < 0) {
    if (rowLower[i] <= -HIGHS_CONST_INF) return false;
    low = rowLower[i];
    upp = rowLower[i];
  } else {
    low = rowLower[i];
    upp = rowUpper[i];
  }

  pair<double, double> p = getBoundsImpliedFree(low, upp, col, i, k);
  low = p.first;
  upp = p.second;

  if (low > -HIGHS_CONST_INF) low = low / Avalue[k];
  if (upp < HIGHS_CONST_INF) upp = upp / Avalue[k];

  // if implied free
  if (colLower[col] <= low && low <= upp && upp <= colUpper[col]) {
    removeImpliedFreeColumn(col, i, k);
    return true;
  }
  // else calculate implied bounds
  else if (colLower[col] <= low && low <= upp) {
    if (implColLower[col] < low) {
      implColLower[col] = low;
      implColUpperRowIndex[col] = i;
    }
    // JAJH(190419): Segfault here since i is a row index and
    // segfaults (on dcp1) due to i=4899 exceeding column dimension of
    // 3007. Hence correction. Also pattern-matches the next case :-)
    //    implColDualUpper[i] = 0;
    implColDualUpper[col] = 0;
  } else if (low <= upp && upp <= colUpper[col]) {
    if (implColUpper[col] > upp) {
      implColUpper[col] = upp;
      implColUpperRowIndex[col] = i;
    }
    implColDualLower[col] = 0;
  }

  return false;
}

// used to remove column too, now possible to just modify bounds
void Presolve::removeRow(int i) {
  hasChange = true;
  flagRow[i] = 0;
  for (int k = ARstart[i]; k < ARstart[i + 1]; ++k) {
    int j = ARindex[k];
    if (flagCol[j]) {
      nzCol[j]--;
      // if now singleton add to list
      if (nzCol[j] == 1) {
        int index = getSingColElementIndexInA(j);
        if (index >= 0)
          singCol.push_back(j);
        else
          cout << "Warning: Column " << j
               << " with 1 nz but not in singCol or? Row removing of " << i
               << ". Ignored.\n";
      }
      // if it was a singleton column remove from list and problem
      if (nzCol[j] == 0) removeEmptyColumn(j);
    }
  }
}

void Presolve::fillStackRowBounds(int row) {
  postValue.push(rowUpper[row]);
  postValue.push(rowLower[row]);
}

pair<double, double> Presolve::getImpliedRowBounds(int row) {
  double g = 0;
  double h = 0;

  int col;
  for (int k = ARstart[row]; k < ARstart[row + 1]; ++k) {
    col = ARindex[k];
    if (flagCol[col]) {
      if (ARvalue[k] < 0) {
        if (colUpper[col] < HIGHS_CONST_INF)
          g += ARvalue[k] * colUpper[col];
        else {
          g = -HIGHS_CONST_INF;
          break;
        }
      } else {
        if (colLower[col] > -HIGHS_CONST_INF)
          g += ARvalue[k] * colLower[col];
        else {
          g = -HIGHS_CONST_INF;
          break;
        }
      }
    }
  }

  for (int k = ARstart[row]; k < ARstart[row + 1]; ++k) {
    col = ARindex[k];
    if (flagCol[col]) {
      if (ARvalue[k] < 0) {
        if (colLower[col] > -HIGHS_CONST_INF)
          h += ARvalue[k] * colLower[col];
        else {
          h = HIGHS_CONST_INF;
          break;
        }
      } else {
        if (colUpper[col] < HIGHS_CONST_INF)
          h += ARvalue[k] * colUpper[col];
        else {
          h = HIGHS_CONST_INF;
          break;
        }
      }
    }
  }
  return make_pair(g, h);
}

void Presolve::setVariablesToBoundForForcingRow(const int row,
                                                const bool isLower) {
  int k, col;
  if (iPrint > 0)
    cout << "PR: Forcing row " << row
         << " removed. Following variables too:   nzRow=" << nzRow[row] << endl;

  flagRow[row] = 0;
  addChange(FORCING_ROW, row, 0);
  k = ARstart[row];
  while (k < ARstart[row + 1]) {
    col = ARindex[k];
    if (flagCol[col]) {
      double value;
      if ((ARvalue[k] < 0 && isLower) || (ARvalue[k] > 0 && !isLower))
        value = colUpper[col];
      else
        value = colLower[col];

      setPrimalValue(col, value);
      valueColDual[col] = colCost[col];
      vector<double> bnds({colLower[col], colUpper[col]});
      oldBounds.push(make_pair(col, bnds));
      addChange(FORCING_ROW_VARIABLE, 0, col);

      if (iPrint > 0)
        cout << "PR:      Variable  " << col << " := " << value << endl;
      countRemovedCols(FORCING_ROW);
    }
    ++k;
  }

  if (nzRow[row] == 1) singRow.remove(row);

  countRemovedRows(FORCING_ROW);
}

void Presolve::dominatedConstraintProcedure(const int i, const double g,
                                            const double h) {
  int j;
  double val;
  if (h < HIGHS_CONST_INF) {
    // fill in implied bounds arrays
    if (h < implRowValueUpper[i]) {
      implRowValueUpper[i] = h;
    }
    if (h <= rowUpper[i]) implRowDualLower[i] = 0;

    // calculate implied bounds for discovering free column singletons
    for (int k = ARstart[i]; k < ARstart[i + 1]; ++k) {
      j = ARindex[k];
      if (flagCol[j]) {
        if (ARvalue[k] < 0 && colLower[j] > -HIGHS_CONST_INF) {
          val = (rowLower[i] - h) / ARvalue[k] + colLower[j];
          if (val < implColUpper[j]) {
            implColUpper[j] = val;
            implColUpperRowIndex[j] = i;
          }
        } else if (ARvalue[k] > 0 && colUpper[j] < HIGHS_CONST_INF) {
          val = (rowLower[i] - h) / ARvalue[k] + colUpper[j];
          if (val > implColLower[j]) {
            implColLower[j] = val;
            implColLowerRowIndex[j] = i;
          }
        }
      }
    }
  }
  if (g > -HIGHS_CONST_INF) {
    // fill in implied bounds arrays
    if (g > implRowValueLower[i]) {
      implRowValueLower[i] = g;
    }
    if (g >= rowLower[i]) implRowDualUpper[i] = 0;

    // calculate implied bounds for discovering free column singletons
    for (int k = ARstart[i]; k < ARstart[i + 1]; ++k) {
      int j = ARindex[k];
      if (flagCol[j]) {
        if (ARvalue[k] < 0 && colUpper[j] < HIGHS_CONST_INF) {
          val = (rowUpper[i] - g) / ARvalue[k] + colUpper[j];
          if (val > implColLower[j]) {
            implColLower[j] = val;
            implColLowerRowIndex[j] = i;
          }
        } else if (ARvalue[k] > 0 && colLower[j] > -HIGHS_CONST_INF) {
          val = (rowUpper[i] - g) / ARvalue[k] + colLower[j];
          if (val < implColUpper[j]) {
            implColUpper[j] = val;
            implColUpperRowIndex[j] = i;
          }
        }
      }
    }
  }
}

void Presolve::removeForcingConstraints(int mainIter) {
  double g, h;
  pair<double, double> implBounds;

  for (int i = 0; i < numRow; ++i)
    if (flagRow[i]) {
      if (nzRow[i] == 0) {
        removeEmptyRow(i);
        countRemovedRows(EMPTY_ROW);
        continue;
      }

      // removeRowSingletons will handle just after removeForcingConstraints
      if (nzRow[i] == 1) continue;

      timer.recordStart(FORCING_ROW);
      implBounds = getImpliedRowBounds(i);

      g = implBounds.first;
      h = implBounds.second;

      // Infeasible row
      if (g > rowUpper[i] || h < rowLower[i]) {
        if (iPrint > 0) cout << "PR: Problem infeasible." << endl;
        status = Infeasible;
        timer.recordFinish(FORCING_ROW);
        return;
      }
      // Forcing row
      else if (g == rowUpper[i]) {
        setVariablesToBoundForForcingRow(i, true);
      } else if (h == rowLower[i]) {
        setVariablesToBoundForForcingRow(i, false);
      }
      // Redundant row
      else if (g >= rowLower[i] && h <= rowUpper[i]) {
        removeRow(i);
        addChange(REDUNDANT_ROW, i, 0);
        if (iPrint > 0)
          cout << "PR: Redundant row " << i << " removed." << endl;
        countRemovedRows(REDUNDANT_ROW);
      }
      // Dominated constraints
      else {
        timer.recordFinish(FORCING_ROW);
        timer.recordStart(DOMINATED_ROW_BOUNDS);
        dominatedConstraintProcedure(i, g, h);
        timer.recordFinish(DOMINATED_ROW_BOUNDS);
        continue;
      }
      timer.recordFinish(FORCING_ROW);
    }
  if (mainIter) {
  }  // surpress warning.
}

void Presolve::removeRowSingletons() {
  timer.recordStart(SING_ROW);
  int i;
  int singRowZ = singRow.size();
  /*
  if (singRowZ == 36) {
    printf("JAJH: singRow.size() = %d\n", singRowZ);fflush(stdout);
  }
  */
  while (!(singRow.empty())) {
    i = singRow.front();
    singRow.pop_front();

    assert(flagRow[i]);

    int k = getSingRowElementIndexInAR(i);
    // JAJH(190419): This throws a segfault with greenbea and greenbeb since
    // k=-1
    if (k < 0) {
      printf("In removeRowSingletons: %d = k < 0\n", k);
      printf("   Occurs for case when initial singRow.size() = %d\n", singRowZ);
      fflush(stdout);
    }
    int j = ARindex[k];

    // add old bounds OF X to checker and for postsolve
    if (iKKTcheck == 1) {
      vector<pair<int, double>> bndsL, bndsU, costS;
      bndsL.push_back(make_pair(j, colLower[j]));
      bndsU.push_back(make_pair(j, colUpper[j]));
      chk.cLowers.push(bndsL);
      chk.cUppers.push(bndsU);
    }

    vector<double> bnds({colLower[j], colUpper[j], rowLower[i], rowUpper[i]});
    oldBounds.push(make_pair(j, bnds));

    double aij = ARvalue[k];
    /*		//before update bounds of x take it out of rows with implied row
    bounds for (int r = Astart[j]; r<Aend[j]; r++) { if
    (flagRow[Aindex[r]]) { int rr = Aindex[r]; if (implRowValueLower[rr] >
    -HIGHS_CONST_INF) { if (aij > 0) implRowValueLower[rr] =
    implRowValueLower[rr] - aij*colLower[j]; else implRowValueLower[rr] =
    implRowValueLower[rr] - aij*colUpper[j];
                    }
                    if (implRowValueUpper[rr] < HIGHS_CONST_INF) {
                            if (aij > 0)
                                    implRowValueUpper[rr] =
    implRowValueUpper[rr] - aij*colUpper[j]; else implRowValueUpper[rr] =
    implRowValueUpper[rr] - aij*colLower[j];
                    }
            }
    }*/

    // update bounds of X
    if (aij > 0) {
      if (rowLower[i] != -HIGHS_CONST_INF)
        colLower[j] =
            max(max(rowLower[i] / aij, -HIGHS_CONST_INF), colLower[j]);
      if (rowUpper[i] != HIGHS_CONST_INF)
        colUpper[j] = min(min(rowUpper[i] / aij, HIGHS_CONST_INF), colUpper[j]);
    } else if (aij < 0) {
      if (rowLower[i] != -HIGHS_CONST_INF)
        colUpper[j] = min(min(rowLower[i] / aij, HIGHS_CONST_INF), colUpper[j]);
      if (rowUpper[i] != HIGHS_CONST_INF)
        colLower[j] =
            max(max(rowUpper[i] / aij, -HIGHS_CONST_INF), colLower[j]);
    }

    /*		//after update bounds of x add to rows with implied row bounds
    for (int r = Astart[j]; r<Aend[j]; r++) {
            if (flagRow[r]) {
                    int rr = Aindex[r];
                    if (implRowValueLower[rr] > -HIGHS_CONST_INF) {
                            if (aij > 0)
                                    implRowValueLower[rr] =
    implRowValueLower[rr] + aij*colLower[j]; else implRowValueLower[rr] =
    implRowValueLower[rr] + aij*colUpper[j];
                    }
                    if (implRowValueUpper[rr] < HIGHS_CONST_INF) {
                            if (aij > 0)
                                    implRowValueUpper[rr] =
    implRowValueUpper[rr] + aij*colUpper[j]; else implRowValueUpper[rr] =
    implRowValueUpper[rr] + aij*colLower[j];
                    }
            }
    }*/

    // check for feasibility
    if (colLower[j] > colUpper[j] + tol) {
      status = Infeasible;
      timer.recordFinish(SING_ROW);
      return;
    }

    if (iPrint > 0)
      cout << "PR: Singleton row " << i << " removed. Bounds of variable  " << j
           << " modified: l= " << colLower[j] << " u=" << colUpper[j]
           << ", aij = " << aij << endl;

    addChange(SING_ROW, i, j);
    postValue.push(colCost[j]);
    removeRow(i);

    if (flagCol[j] && colLower[j] == colUpper[j]) removeIfFixed(j);

    countRemovedRows(SING_ROW);
  }
  timer.recordFinish(SING_ROW);
}

void Presolve::addChange(PresolveRule type, int row, int col) {
  change ch;
  ch.type = type;
  ch.row = row;
  ch.col = col;
  chng.push(ch);

  if (type < PRESOLVE_RULES_COUNT) timer.addChange(type);
}

// when setting a value to a primal variable and eliminating row update b,
// singleton Rows linked list, number of nonzeros in rows
void Presolve::setPrimalValue(int j, double value) {
  flagCol[j] = 0;
  if (!hasChange) hasChange = true;
  valuePrimal[j] = value;

  // update nonzeros
  for (int k = Astart[j]; k < Aend[j]; ++k) {
    int row = Aindex[k];
    if (flagRow[row]) {
      nzRow[row]--;

      // update singleton row list
      if (nzRow[row] == 1)
        singRow.push_back(row);
      else if (nzRow[row] == 0)
        singRow.remove(row);
    }
  }

  // update values if necessary
  if (fabs(value) > 0) {
    // RHS
    vector<pair<int, double>> bndsL, bndsU;

    for (int k = Astart[j]; k < Aend[j]; ++k)
      if (flagRow[Aindex[k]]) {
        if (iKKTcheck == 1) {
          bndsL.push_back(make_pair(Aindex[k], rowLower[Aindex[k]]));
          bndsU.push_back(make_pair(Aindex[k], rowUpper[Aindex[k]]));
        }
        if (rowLower[Aindex[k]] > -HIGHS_CONST_INF)
          rowLower[Aindex[k]] -= Avalue[k] * value;
        if (rowUpper[Aindex[k]] < HIGHS_CONST_INF)
          rowUpper[Aindex[k]] -= Avalue[k] * value;

        if (implRowValueLower[Aindex[k]] > -HIGHS_CONST_INF)
          implRowValueLower[Aindex[k]] -= Avalue[k] * value;
        if (implRowValueUpper[Aindex[k]] < HIGHS_CONST_INF)
          implRowValueUpper[Aindex[k]] -= Avalue[k] * value;
      }

    if (iKKTcheck == 1) {
      chk.rLowers.push(bndsL);
      chk.rUppers.push(bndsU);
    }

    // shift objective
    if (colCost[j] != 0) objShift += colCost[j] * value;
  }
}

void Presolve::checkForChanges(int iteration) {
  if (iteration <= 2) {
    // flagCol has one more element at end which is zero
    // from removeDoubletonEquatoins, needed for AR matrix manipulation
    if (none_of(flagCol.begin(), flagCol.begin() + numCol,
                [](int i) { return i == 0; }) &&
        none_of(flagRow.begin(), flagRow.begin() + numRow,
                [](int i) { return i == 0; })) {
      if (iPrint > 0)
        cout << "PR: No variables were eliminated at presolve." << endl;
      noPostSolve = true;
      return;
    }
  }
  resizeProblem();
  status = stat::Reduced;
}

// void Presolve::reportTimes() {
//   int reportList[] = {EMPTY_ROW,
//                       FIXED_COL,
//                       SING_ROW,
//                       DOUBLETON_EQUATION,
//                       FORCING_ROW,
//                       REDUNDANT_ROW,
//                       FREE_SING_COL,
//                       SING_COL_DOUBLETON_INEQ,
//                       IMPLIED_FREE_SING_COL,
//                       DOMINATED_COLS,
//                       WEAKLY_DOMINATED_COLS};
//   int reportCount = sizeof(reportList) / sizeof(int);

//   printf("Presolve rules ");
//   for (int i = 0; i < reportCount; ++i) {
//     printf(" %s", timer.itemNames[reportList[i]].c_str());
//     cout << flush;
//   }

//   printf("\n");
//   cout << "Time spent     " << flush;
//   for (int i = 0; i < reportCount; ++i) {
//     float f = (float)timer.itemTicks[reportList[i]];
//     if (f < 0.01)
//       cout << setw(4) << " <.01 ";
//     else
//       printf(" %3.2f ", f);
//   }
//   printf("\n");
// }

// void Presolve::recordCounts(const string fileName) {
//   ofstream myfile;
//   myfile.open(fileName.c_str(), ios::app);
//   int reportList[] = {EMPTY_ROW,
//                       FIXED_COL,
//                       SING_ROW,
//                       DOUBLETON_EQUATION,
//                       FORCING_ROW,
//                       REDUNDANT_ROW,
//                       FREE_SING_COL,
//                       SING_COL_DOUBLETON_INEQ,
//                       IMPLIED_FREE_SING_COL,
//                       DOMINATED_COLS,
//                       WEAKLY_DOMINATED_COLS,
//                       EMPTY_COL};
//   int reportCount = sizeof(reportList) / sizeof(int);

//   myfile << "Problem " << modelName << ":\n";
//   myfile << "Rule   , removed rows , removed cols , time  \n";

//   int cRows = 0, cCols = 0;
//   for (int i = 0; i < reportCount; ++i) {
//     float f = (float)timer.itemTicks[reportList[i]];

//     myfile << setw(7) << timer.itemNames[reportList[i]].c_str() << ", "
//            << setw(7) << countRemovedRows[reportList[i]] << ", " << setw(7)
//            << countRemovedCols[reportList[i]] << ", ";
//     if (f < 0.001)
//       myfile << setw(7) << " <.001 ";
//     else
//       myfile << setw(7) << setprecision(3) << f;
//     myfile << endl;

//     cRows += countRemovedRows[reportList[i]];
//     cCols += countRemovedCols[reportList[i]];
//   }

//   if (!noPostSolve) {
//     if (cRows != numRowOriginal - numRow) cout << "Wrong row reduction
//     count\n"; if (cCols != numColOriginal - numCol) cout << "Wrong col
//     reduction count\n";

//     myfile << setw(7) << "Total "
//            << ", " << setw(7) << numRowOriginal - numRow << ", " << setw(7)
//            << numColOriginal - numCol;
//   } else {
//     myfile << setw(7) << "Total "
//            << ", " << setw(7) << 0 << ", " << setw(7) << 0;
//   }
//   myfile << endl << " \\\\ " << endl;
//   myfile.close();
// }

void Presolve::resizeImpliedBounds() {
  // implied bounds for crashes
  // row duals
  vector<double> temp = implRowDualLower;
  vector<double> teup = implRowDualUpper;
  implRowDualLower.resize(numRow);
  implRowDualUpper.resize(numRow);

  int k = 0;
  for (int i = 0; i < numRowOriginal; ++i)
    if (flagRow[i]) {
      implRowDualLower[k] = temp[i];
      implRowDualUpper[k] = teup[i];
      k++;
    }

  // row value
  temp = implRowValueLower;
  teup = implRowValueUpper;
  implRowValueLower.resize(numRow);
  implRowValueUpper.resize(numRow);
  k = 0;
  for (int i = 0; i < numRowOriginal; ++i)
    if (flagRow[i]) {
      if (temp[i] < rowLower[i]) temp[i] = rowLower[i];
      implRowValueLower[k] = temp[i];
      if (teup[i] > rowUpper[i]) teup[i] = rowUpper[i];
      implRowValueUpper[k] = teup[i];
      k++;
    }

  // column dual
  temp = implColDualLower;
  teup = implColDualUpper;
  implColDualLower.resize(numCol);
  implColDualUpper.resize(numCol);

  k = 0;
  for (int i = 0; i < numColOriginal; ++i)
    if (flagCol[i]) {
      implColDualLower[k] = temp[i];
      implColDualUpper[k] = teup[i];
      k++;
    }

  // column value
  temp = implColLower;
  teup = implColUpper;
  implColLower.resize(numCol);
  implColUpper.resize(numCol);

  k = 0;
  for (int i = 0; i < numColOriginal; ++i)
    if (flagCol[i]) {
      if (temp[i] < colLower[i]) temp[i] = colLower[i];
      implColLower[k] = temp[i];
      if (teup[i] > colUpper[i]) teup[i] = colUpper[i];
      implColUpper[k] = teup[i];
      k++;
    }
}

int Presolve::getSingRowElementIndexInAR(int i) {
  int k = ARstart[i];
  while (!flagCol[ARindex[k]]) ++k;
  if (k >= ARstart[i + 1]) {
    cout << "Error during presolve: no variable found in singleton row " << i
         << endl;
    return -1;
  }
  int rest = k + 1;
  while (rest < ARstart[i + 1] && !flagCol[ARindex[rest]]) ++rest;
  if (rest < ARstart[i + 1]) {
    cout << "Error during presolve: more variables found in singleton row " << i
         << endl;
    return -1;
  }
  return k;
}

int Presolve::getSingColElementIndexInA(int j) {
  int k = Astart[j];
  while (!flagRow[Aindex[k]]) ++k;
  if (k >= Aend[j]) {
    cout << "Error during presolve: no variable found in singleton col " << j
         << ".";
    return -1;
  }
  int rest = k + 1;
  while (rest < Aend[j] && !flagRow[Aindex[rest]]) ++rest;
  if (rest < Aend[j]) {
    cout << "Error during presolve: more variables found in singleton col " << j
         << ".";
    return -1;
  }
  return k;
}

void Presolve::testAnAR(int post) {
  int rows = numRow;
  int cols = numCol;
  int i, j, k;

  double valueA = 0;
  double valueAR = 0;
  bool hasValueA, hasValueAR;

  if (post) {
    rows = numRowOriginal;
    cols = numColOriginal;
  }

  // check that A = AR
  for (i = 0; i < rows; ++i) {
    for (j = 0; j < cols; ++j) {
      if (post == 0)
        if (!flagRow[i] || !flagCol[j]) continue;
      hasValueA = false;
      for (k = Astart[j]; k < Aend[j]; ++k)
        if (Aindex[k] == i) {
          hasValueA = true;
          valueA = Avalue[k];
        }

      hasValueAR = false;
      for (k = ARstart[i]; k < ARstart[i + 1]; ++k)
        if (ARindex[k] == j) {
          hasValueAR = true;
          valueAR = ARvalue[k];
        }

      if (hasValueA != hasValueAR)
        cout << "    MATRIX is0   DIFF row=" << i << " col=" << j
             << "           ------------A: " << hasValueA
             << "  AR: " << hasValueAR << endl;
      else if (hasValueA && valueA != valueAR)
        cout << "    MATRIX VAL  DIFF row=" << i << " col=" << j
             << "           ------------A: " << valueA << "  AR: " << valueAR
             << endl;
    }
  }

  if (post == 0) {
    // check nz
    int nz = 0;
    for (i = 0; i < rows; ++i) {
      if (!flagRow[i]) continue;
      nz = 0;
      for (k = ARstart[i]; k < ARstart[i + 1]; ++k)
        if (flagCol[ARindex[k]]) nz++;
      if (nz != nzRow[i])
        cout << "    NZ ROW      DIFF row=" << i << " nzRow=" << nzRow[i]
             << " actually " << nz << "------------" << endl;
    }

    for (j = 0; j < cols; ++j) {
      if (!flagCol[j]) continue;
      nz = 0;
      for (k = Astart[j]; k < Aend[j]; ++k)
        if (flagRow[Aindex[k]]) nz++;
      if (nz != nzCol[j])
        cout << "    NZ COL      DIFF col=" << j << " nzCol=" << nzCol[j]
             << " actually " << nz << "------------" << endl;
    }
  }
}

// todo: error reporting.
HighsPostsolveStatus Presolve::postsolve(const HighsSolution& reduced_solution,
                                         HighsSolution& recovered_solution) {
  colValue = reduced_solution.col_value;
  colDual = reduced_solution.col_dual;
  rowDual = reduced_solution.row_dual;

  // todo: add nonbasic flag to Solution.
  // todo: change to new basis info structure later or keep.
  // basis info and solution should be somehow connected to each other.

  // here noPostSolve is always false. If the problem has not been reduced
  // Presolve::postsolve(..) is never called. todo: delete block below. For now
  // left just as legacy.
  // if (noPostSolve) {
  //   // set valuePrimal
  //   for (int i = 0; i < numCol; ++i) {
  //     valuePrimal[i] = colValue[i];
  //     valueColDual[i] = colDual[i];
  //   }
  //   for (int i = 0; i < numRow; ++i) valueRowDual[i] = rowDual[i];
  //   // For KKT check: first check solverz` results before we do any postsolve
  //   if (iKKTcheck == 1) {
  //     chk.passSolution(colValue, colDual, rowDual);
  //     chk.passBasis(col_status, row_status);
  //     chk.makeKKTCheck();
  //   }
  //   // testBasisMatrixSingularity();
  //   return HighsPostsolveStatus::NoPostsolve;
  // }

  // For KKT check: first check solver results before we do any postsolve
  if (iKKTcheck == 1) {
    cout << "----KKT check on HiGHS solution-----\n";

    chk.passSolution(colValue, colDual, rowDual);
    chk.passBasis(col_status, row_status);
    chk.makeKKTCheck();
  }
  // So there have been changes definitely ->
  makeACopy();  // so we can efficiently calculate primal and dual values

  //	iKKTcheck = false;
  // set corresponding parts of solution vectors:
  int j = 0;
  vector<int> eqIndexOfReduced(numCol, -1);
  vector<int> eqIndexOfReduROW(numRow, -1);
  for (int i = 0; i < numColOriginal; ++i)
    if (cIndex[i] > -1) {
      eqIndexOfReduced[j] = i;
      ++j;
    }
  j = 0;
  for (int i = 0; i < numRowOriginal; ++i)
    if (rIndex[i] > -1) {
      eqIndexOfReduROW[j] = i;
      ++j;
    }

  vector<HighsBasisStatus> temp_col_status = col_status;
  vector<HighsBasisStatus> temp_row_status = row_status;

  nonbasicFlag.assign(numColOriginal + numRowOriginal, 1);
  col_status.assign(numColOriginal, HighsBasisStatus::NONBASIC);  // Was LOWER
  row_status.assign(numRowOriginal, HighsBasisStatus::NONBASIC);  // Was LOWER

  for (int i = 0; i < numCol; ++i) {
    int iCol = eqIndexOfReduced[i];
    assert(iCol < (int)valuePrimal.size());
    assert(iCol < (int)valueColDual.size());
    assert(iCol >= 0);
    valuePrimal[iCol] = colValue[i];
    valueColDual[iCol] = colDual[i];
    col_status[iCol] = temp_col_status[i];
  }

  for (int i = 0; i < numRow; ++i) {
    int iRow = eqIndexOfReduROW[i];
    valueRowDual[iRow] = rowDual[i];
    row_status[iRow] = temp_row_status[i];
  }

  // cmpNBF(-1, -1);

  double z;
  vector<int> fRjs;
  while (!chng.empty()) {
    change c = chng.top();
    chng.pop();
    // cout<<"chng.pop:       "<<c.col<<"       "<<c.row << endl;

    setBasisElement(c);
    if (iKKTcheck == 1) chk.replaceBasis(col_status, row_status);

    switch (c.type) {
      case DOUBLETON_EQUATION: {  // Doubleton equation row
        getDualsDoubletonEquation(c.row, c.col);

        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout
                << "----KKT check after doubleton equation re-introduced. Row: "
                << c.row << ", column " << c.col << " -----\n";
          chk.addChange(17, c.row, c.col, valuePrimal[c.col],
                        valueColDual[c.col], valueRowDual[c.row]);
          chk.replaceBasis(col_status, row_status);
          chk.makeKKTCheck();
        }
        // exit(2);
        break;
      }
      case DOUBLETON_EQUATION_ROW_BOUNDS_UPDATE: {
        // new bounds from doubleton equation, retrieve old ones
        // just for KKT check, not called otherwise
        chk.addChange(171, c.row, c.col, 0, 0, 0);
        break;
      }
      case DOUBLETON_EQUATION_NEW_X_NONZERO: {
        // matrix transformation from doubleton equation, case x still there
        // case new x is not 0
        // just change value of entry in row for x

        int indi;
        for (indi = ARstart[c.row]; indi < ARstart[c.row + 1]; ++indi)
          if (ARindex[indi] == c.col) break;
        ARvalue[indi] = postValue.top();
        for (indi = Astart[c.col]; indi < Aend[c.col]; ++indi)
          if (Aindex[indi] == c.row) break;
        Avalue[indi] = postValue.top();

        if (iKKTcheck == 1)
          chk.addChange(172, c.row, c.col, postValue.top(), 0, 0);
        postValue.pop();

        break;
      }
      case DOUBLETON_EQUATION_X_ZERO_INITIALLY: {
        // matrix transformation from doubleton equation, retrieve old value
        // case when row does not have x initially: entries for row i swap x and
        // y cols

        int indi, yindex;
        yindex = (int)postValue.top();
        postValue.pop();

        // reverse AR for case when x is zero and y entry has moved
        for (indi = ARstart[c.row]; indi < ARstart[c.row + 1]; ++indi)
          if (ARindex[indi] == c.col) break;
        ARvalue[indi] = postValue.top();
        ARindex[indi] = yindex;

        // reverse A for case when x is zero and y entry has moved
        for (indi = Astart[c.col]; indi < Aend[c.col]; ++indi)
          if (Aindex[indi] == c.row) break;

        // recover x: column decreases by 1
        // if indi is not Aend-1 swap elements indi and Aend-1
        if (indi != Aend[c.col] - 1) {
          double tmp = Avalue[Aend[c.col] - 1];
          int tmpi = Aindex[Aend[c.col] - 1];
          Avalue[Aend[c.col] - 1] = Avalue[indi];
          Aindex[Aend[c.col] - 1] = Aindex[indi];
          Avalue[indi] = tmp;
          Aindex[indi] = tmpi;
        }
        Aend[c.col]--;

        // recover y: column increases by 1
        // update A: append X column to end of array
        int st = Avalue.size();
        for (int ind = Astart[yindex]; ind < Aend[yindex]; ++ind) {
          Avalue.push_back(Avalue[ind]);
          Aindex.push_back(Aindex[ind]);
        }
        Avalue.push_back(postValue.top());
        Aindex.push_back(c.row);
        Astart[yindex] = st;
        Aend[yindex] = Avalue.size();

        if (iKKTcheck == 1)
          chk.addChange(173, c.row, c.col, postValue.top(), (double)yindex, 0);
        postValue.pop();

        break;
      }
      case DOUBLETON_EQUATION_NEW_X_ZERO_AR_UPDATE: {
        // sp case x disappears row representation change
        int indi;
        for (indi = ARstart[c.row]; indi < ARstart[c.row + 1]; ++indi)
          if (ARindex[indi] == numColOriginal) break;
        ARindex[indi] = c.col;
        ARvalue[indi] = postValue.top();

        if (iKKTcheck == 1) {
          chk.ARindex[indi] = c.col;
          chk.ARvalue[indi] = postValue.top();
        }

        postValue.pop();

        break;
      }
      case DOUBLETON_EQUATION_NEW_X_ZERO_A_UPDATE: {
        // sp case x disappears column representation change
        // here A is copied from AR array at end of presolve so need to expand x
        // column  Aend[c.col]++; wouldn't do because old value is overriden
        double oldXvalue = postValue.top();
        postValue.pop();
        int x = c.col;

        // update A: append X column to end of array
        int st = Avalue.size();
        for (int ind = Astart[x]; ind < Aend[x]; ++ind) {
          Avalue.push_back(Avalue[ind]);
          Aindex.push_back(Aindex[ind]);
        }
        Avalue.push_back(oldXvalue);
        Aindex.push_back(c.row);
        Astart[x] = st;
        Aend[x] = Avalue.size();

        break;
      }
      case EMPTY_ROW: {
        valueRowDual[c.row] = 0;
        flagRow[c.row] = 1;
        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after empty row " << c.row
                 << " re-introduced-----\n";
          chk.addChange(0, c.row, 0, 0, 0, 0);
          chk.makeKKTCheck();
        }
        break;
      }
      case SING_ROW: {
        // valuePrimal is already set for this one, colDual also, we need
        // rowDual. AR copy keeps full matrix.  col dual maybe infeasible, we
        // need to check.  recover old bounds and see
        getDualsSingletonRow(c.row, c.col);

        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after singleton row " << c.row
                 << " re-introduced. Variable: " << c.col << " -----\n";
          chk.addChange(1, c.row, c.col, valuePrimal[c.col],
                        valueColDual[c.col], valueRowDual[c.row]);
          chk.replaceBasis(col_status, row_status);
          chk.makeKKTCheck();
        }
        break;
      }
      case FORCING_ROW_VARIABLE:
        fRjs.push_back(c.col);
        flagCol[c.col] = 1;
        if (iKKTcheck == 1 && valuePrimal[c.col] != 0)
          chk.addChange(22, c.row, c.col, 0, 0, 0);
        break;
      case FORCING_ROW: {
        string str = getDualsForcingRow(c.row, fRjs);

        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after forcing row " << c.row
                 << " re-introduced. Variable(s): " << str << " -----\n";
          chk.replaceBasis(col_status, row_status);
          chk.addChange(3, c.row, 0, 0, 0, valueRowDual[c.row]);
          chk.makeKKTCheck();
        }
        fRjs.clear();
        break;
      }
      case REDUNDANT_ROW: {
        // this is not zero if the row bounds got relaxed and transferred to a
        // column which then had a nonzero dual.
        valueRowDual[c.row] = 0;

        flagRow[c.row] = 1;

        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after redundant row " << c.row
                 << " re-introduced.----------------\n";
          chk.addChange(0, c.row, 0, 0, 0, 0);
          chk.makeKKTCheck();
        }
        break;
      }
      case FREE_SING_COL:
      case IMPLIED_FREE_SING_COL: {
        // colDual rowDual already set.
        // calculate row value without xj
        double aij = getaij(c.row, c.col);
        double sum = 0;
        for (int k = ARstart[c.row]; k < ARstart[c.row + 1]; ++k)
          if (flagCol[ARindex[k]]) sum += valuePrimal[ARindex[k]] * ARvalue[k];

        double rowlb = postValue.top();
        postValue.pop();
        double rowub = postValue.top();
        postValue.pop();

        // calculate xj
        if (valueRowDual[c.row] < 0) {
          // row is at lower bound
          valuePrimal[c.col] = (rowlb - sum) / aij;
        } else if (valueRowDual[c.row] > 0) {
          // row is at upper bound
          valuePrimal[c.col] = (rowub - sum) / aij;
        } else if (rowlb == rowub)
          valuePrimal[c.col] = (rowlb - sum) / aij;
        else if (colCostAtEl[c.col] > 0) {
          // we are interested in the lowest possible value of x:
          // max { l_j, bound implied by row i }
          double bndL;
          if (aij > 0)
            bndL = (rowlb - sum) / aij;
          else
            bndL = (rowub - sum) / aij;
          valuePrimal[c.col] = max(colLowerOriginal[c.col], bndL);
        } else if (colCostAtEl[c.col] < 0) {
          // we are interested in the highest possible value of x:
          // min { u_j, bound implied by row i }
          double bndU;
          if (aij < 0)
            bndU = (rowlb - sum) / aij;
          else
            bndU = (rowub - sum) / aij;
          valuePrimal[c.col] = min(colUpperOriginal[c.col], bndU);
        } else {  // cost is zero
          double bndL, bndU;
          if (aij > 0) {
            bndL = (rowlb - sum) / aij;
            bndU = (rowub - sum) / aij;
          } else {
            bndL = (rowub - sum) / aij;
            bndU = (rowlb - sum) / aij;
          }
          double valuePrimalUB = min(colUpperOriginal[c.col], bndU);
          double valuePrimalLB = max(colLowerOriginal[c.col], bndL);
          if (valuePrimalUB < valuePrimalLB - tol) {
            cout << "Postsolve error: inconsistent bounds for implied free "
                    "column singleton "
                 << c.col << endl;
          }

          if (fabs(valuePrimalLB) < fabs(valuePrimalUB))
            valuePrimal[c.col] = valuePrimalLB;
          else
            valuePrimal[c.col] = valuePrimalUB;
        }
        sum = sum + valuePrimal[c.col] * aij;

        double costAtTimeOfElimination = postValue.top();
        postValue.pop();
        objShift += (costAtTimeOfElimination * sum) / aij;

        flagRow[c.row] = 1;
        flagCol[c.col] = 1;
        // valueRowDual[c.row] = 0;

        if (iKKTcheck == 1) {
          chk.addCost(c.col, costAtTimeOfElimination);
          if (c.type == FREE_SING_COL && chk.print == 1)
            cout << "----KKT check after free col singleton " << c.col
                 << " re-introduced. Row: " << c.row << " -----\n";
          else if (c.type == IMPLIED_FREE_SING_COL && chk.print == 1)
            cout << "----KKT check after implied free col singleton " << c.col
                 << " re-introduced. Row: " << c.row << " -----\n";
          chk.addChange(4, c.row, c.col, valuePrimal[c.col],
                        valueColDual[c.col], valueRowDual[c.row]);
          chk.makeKKTCheck();
        }
        break;
      }
      case SING_COL_DOUBLETON_INEQ: {
        // column singleton in a doubleton equation.
        // colDual already set. need valuePrimal from stack. maybe change
        // rowDual depending on bounds. old bounds kept in oldBounds. variables
        // j,k : we eliminated j and are left with changed bounds on k and no
        // row. c.col is column COL (K) - eliminated, j is with new bounds
        pair<int, vector<double>> p = oldBounds.top();
        oldBounds.pop();
        vector<double> v = get<1>(p);
        int j = get<0>(p);
        // double ubNew = v[1];
        // double lbNew = v[0];
        double cjNew = v[2];
        p = oldBounds.top();
        oldBounds.pop();
        v = get<1>(p);
        double ubOld = v[1];
        double lbOld = v[0];
        double cjOld = v[2];
        p = oldBounds.top();
        oldBounds.pop();
        v = get<1>(p);
        double ubCOL = v[1];
        double lbCOL = v[0];
        double ck = v[2];

        double rowlb = postValue.top();
        postValue.pop();
        double rowub = postValue.top();
        postValue.pop();
        double aik = postValue.top();
        postValue.pop();
        double aij = postValue.top();
        postValue.pop();
        double xj = valuePrimal[j];

        // calculate xk, depending on signs of coeff and cost
        double upp = HIGHS_CONST_INF;
        double low = -HIGHS_CONST_INF;

        if ((aij > 0 && aik > 0) || (aij < 0 && aik < 0)) {
          upp = (rowub - aij * xj) / aik;
          low = (rowlb - aij * xj) / aik;
        } else {
          upp = (rowub - aij * xj) / aik;
          low = (rowlb - aij * xj) / aik;
        }

        double xkValue = 0;
        if (ck == 0) {
          if (low < 0 && upp > 0)
            xkValue = 0;
          else if (fabs(low) < fabs(upp))
            xkValue = low;
          else
            xkValue = upp;
        }

        else if ((ck > 0 && aik > 0) || (ck < 0 && aik < 0)) {
          if (low <= -HIGHS_CONST_INF)
            cout << "ERROR UNBOUNDED? unnecessary check";
          xkValue = low;
        } else if ((ck > 0 && aik < 0) || (ck < 0 && aik > 0)) {
          if (upp >= HIGHS_CONST_INF)
            cout << "ERROR UNBOUNDED? unnecessary check";
          xkValue = upp;
        }

        // primal value and objective shift
        valuePrimal[c.col] = xkValue;
        objShift += -cjNew * xj + cjOld * xj + ck * xkValue;

        // fix duals

        double rowVal = aij * xj + aik * xkValue;
        if (rowub - rowVal > tol && rowVal - rowlb > tol) {
          row_status[c.row] = HighsBasisStatus::BASIC;
          col_status[c.col] = HighsBasisStatus::NONBASIC;
          valueRowDual[c.row] = 0;
          flagRow[c.row] = 1;
          valueColDual[c.col] = getColumnDualPost(c.col);
        } else {
          double lo, up;
          if (fabs(rowlb - rowub) < tol) {
            lo = -HIGHS_CONST_INF;
            up = HIGHS_CONST_INF;
          } else if (fabs(rowub - rowVal) <= tol) {
            lo = 0;
            up = HIGHS_CONST_INF;
          } else if (fabs(rowlb - rowVal) <= tol) {
            lo = -HIGHS_CONST_INF;
            ;
            up = 0;
          }

          colCostAtEl[j] = cjOld;  // revert cost before calculating duals
          getBoundOnLByZj(c.row, j, &lo, &up, lbOld, ubOld);
          getBoundOnLByZj(c.row, c.col, &lo, &up, lbCOL, ubCOL);

          // calculate yi
          if (lo - up > tol)
            cout << "PR: Error in postsolving doubleton inequality " << c.row
                 << " : inconsistent bounds for its dual value.\n";

          // WARNING: bound_row_dual not used. commented out to surpress warning
          // but maybe this causes trouble. Look into when you do dual postsolve
          // again (todo)
          //
          //
          // double bound_row_dual = 0;
          // if (lo > 0) {
          //   bound_row_dual = lo;
          // } else if (up < 0) {
          //   bound_row_dual = up;
          // }

          if (lo > 0 || up < 0) {
            // row is nonbasic, since dual value zero for it is infeasible.
            row_status[c.row] = HighsBasisStatus::NONBASIC;
            col_status[c.col] = HighsBasisStatus::BASIC;
            valueColDual[c.col] = 0;
            flagRow[c.row] = 1;
            valueRowDual[c.row] = getRowDualPost(c.row, c.col);
            valueColDual[j] = getColumnDualPost(j);
          } else {
            // zero row dual is feasible, set row to basic and column to
            // nonbasic.
            row_status[c.row] = HighsBasisStatus::BASIC;
            col_status[c.col] = HighsBasisStatus::NONBASIC;
            valueRowDual[c.row] = 0;
            flagRow[c.row] = 1;
            valueColDual[c.col] = getColumnDualPost(c.col);
          }

          if (iKKTcheck == 1) chk.colDual[j] = valueColDual[j];
        }

        flagCol[c.col] = 1;

        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after col singleton " << c.col
                 << " in doubleton eq re-introduced. Row: " << c.row
                 << " -----\n";

          chk.addChange(5, c.row, c.col, valuePrimal[c.col],
                        valueColDual[c.col], valueRowDual[c.row]);
          chk.replaceBasis(col_status, row_status);
          chk.makeKKTCheck();
        }
        // exit(2);
        break;
      }
      case EMPTY_COL:
      case DOMINATED_COLS:
      case WEAKLY_DOMINATED_COLS: {
        // got valuePrimal, need colDual
        if (c.type != EMPTY_COL) {
          z = colCostAtEl[c.col];
          for (int k = Astart[c.col]; k < Astart[c.col + 1]; ++k)
            if (flagRow[Aindex[k]]) z = z + valueRowDual[Aindex[k]] * Avalue[k];
          valueColDual[c.col] = z;
        }

        flagCol[c.col] = 1;
        if (iKKTcheck == 1) {
          if (c.type == EMPTY_COL && chk.print == 1)
            cout << "----KKT check after empty column " << c.col
                 << " re-introduced.-----------\n";
          else if (c.type == DOMINATED_COLS && chk.print == 1)
            cout << "----KKT check after dominated column " << c.col
                 << " re-introduced.-----------\n";
          else if (c.type == WEAKLY_DOMINATED_COLS && chk.print == 1)
            cout << "----KKT check after weakly dominated column " << c.col
                 << " re-introduced.-----------\n";

          chk.addChange(6, 0, c.col, valuePrimal[c.col], valueColDual[c.col],
                        0);
          chk.makeKKTCheck();
        }
        break;
      }

      case FIXED_COL: {
        // got valuePrimal, need colDual
        valueColDual[c.col] = getColumnDualPost(c.col);

        flagCol[c.col] = 1;
        if (iKKTcheck == 1) {
          if (chk.print == 1)
            cout << "----KKT check after fixed variable " << c.col
                 << " re-introduced.-----------\n";
          chk.addChange(7, 0, c.col, valuePrimal[c.col], valueColDual[c.col],
                        0);
          chk.makeKKTCheck();
        }
        break;
      }
    }
    // cmpNBF(c.row, c.col);
  }

  // cmpNBF();

  // Check number of basic variables
  int num_basic_var = 0;
  for (int iCol = 0; iCol < numColOriginal; iCol++) {
    if (col_status[iCol] == HighsBasisStatus::BASIC) {
      assert(num_basic_var < numRowOriginal);
      if (num_basic_var == numRowOriginal) {
        printf("Error in postsolve: more basic variables than rows\n");
        break;
      }
      num_basic_var++;
    }
  }
  for (int iRow = 0; iRow < numRowOriginal; iRow++) {
    // int iVar = numColOriginal + iRow;
    if (row_status[iRow] == HighsBasisStatus::BASIC) {
      assert(num_basic_var < numRowOriginal);
      if (num_basic_var == numRowOriginal) {
        printf("Error from postsolve: more basic variables than rows\n");
        break;
      }
      num_basic_var++;
    }
  }
  // Return error if the number of basic variables does not equal the
  // number of rows in the original LP
  assert(num_basic_var == numRowOriginal);
  if (num_basic_var != numRowOriginal) {
    printf(
        "Error from postsolve: number of basic variables = %d != %d = number "
        "of rows\n",
        num_basic_var, numRowOriginal);
    return HighsPostsolveStatus::BasisError;
  }

  // cout<<"Singularity check at end of postsolve: ";
  // testBasisMatrixSingularity();

  if (iKKTcheck == 2) {
    if (chk.print == 3) chk.print = 2;
    chk.passSolution(valuePrimal, valueColDual, valueRowDual);
    chk.makeKKTCheck();
  }

  // now recover original model data to pass back to HiGHS
  // A is already recovered!
  // however, A is expressed in terms of Astart, Aend and columns are in
  // different order so
  makeACopy();

  numRow = numRowOriginal;
  numCol = numColOriginal;
  numTot = numRow + numCol;

  rowUpper = rowUpperOriginal;
  rowLower = rowLowerOriginal;

  colUpper = colUpperOriginal;
  colLower = colLowerOriginal;

  colCost = colCostOriginal;

  /*
  nonbasicMove.resize(numTot, 0);
  for (int i = 0; i < numColOriginal; ++i) {
    if (colLower[i] != colUpper[i] && colLower[i] != -HIGHS_CONST_INF)
      nonbasicMove[i] = 1;
    else if (colUpper[i] != HIGHS_CONST_INF)
      nonbasicMove[i] = -1;
    else
      nonbasicMove[i] = 0;
  }
  */
  colValue = valuePrimal;
  colDual = valueColDual;
  rowDual = valueRowDual;
  rowValue.assign(numRow, 0);
  for (int i = 0; i < numRowOriginal; ++i) {
    for (int k = ARstart[i]; k < ARstart[i + 1]; ++k)
      rowValue[i] += valuePrimal[ARindex[k]] * ARvalue[k];
  }
  // JAJH(120519) Added following four lines so that recovered solution is
  // returned
  recovered_solution.col_value = colValue;
  recovered_solution.col_dual = colDual;
  recovered_solution.row_value = rowValue;
  recovered_solution.row_dual = rowDual;
  return HighsPostsolveStatus::SolutionRecovered;
}

void Presolve::setBasisElement(change c) {
  // col_status starts off as [numCol] and has already been increased to
  // [numColOriginal] and row_status starts off as [numRow] and has already been
  // increased to [numRowOriginal] so fill fill in gaps in both

  switch (c.type) {
    case EMPTY_ROW: {
      if (report_postsolve) {
        printf("2.1 : Recover row %3d as %3d (basic): empty row\n", c.row,
               numColOriginal + c.row);
      }
      row_status[c.row] = HighsBasisStatus::BASIC;
      break;
    }
    case SING_ROW:
    case FORCING_ROW_VARIABLE:
    case FORCING_ROW:
    case SING_COL_DOUBLETON_INEQ:
      break;
    case REDUNDANT_ROW: {
      if (report_postsolve) {
        printf("2.3 : Recover row %3d as %3d (basic): redundant\n", c.row,
               numColOriginal + c.row);
      }
      row_status[c.row] = HighsBasisStatus::BASIC;
      break;
    }
    case FREE_SING_COL:
    case IMPLIED_FREE_SING_COL: {
      if (report_postsolve) {
        printf(
            "2.4a: Recover col %3d as %3d (basic): implied free singleton "
            "column\n",
            c.col, numColOriginal + c.row);
      }
      col_status[c.col] = HighsBasisStatus::BASIC;

      if (report_postsolve) {
        printf(
            "2.5b: Recover row %3d as %3d (nonbasic): implied free singleton "
            "column\n",
            c.row, numColOriginal + c.row);
      }
      row_status[c.row] = HighsBasisStatus::NONBASIC;  // Was LOWER
      break;
    }
    case EMPTY_COL:
    case DOMINATED_COLS:
    case WEAKLY_DOMINATED_COLS: {
      if (report_postsolve) {
        printf("2.7 : Recover column %3d (nonbasic): weakly dominated column\n",
               c.col);
      }
      col_status[c.col] = HighsBasisStatus::NONBASIC;  // Was LOWER
      break;
    }
    case FIXED_COL: {  // fixed variable:
      // check if it was NOT after singRow
      if (chng.size() > 0)
        if (chng.top().type != SING_ROW) {
          if (report_postsolve) {
            printf(
                "2.8 : Recover column %3d (nonbasic): weakly dominated "
                "column\n",
                c.col);
          }
          col_status[c.col] = HighsBasisStatus::NONBASIC;  // Was LOWER
        }
      break;
    }
  }
}

/* testing and dev
int Presolve::testBasisMatrixSingularity() {

        HFactor factor;

        //resize matrix in M so we can pass to factor
        int i, j, k;
        int nz = 0;
        int nR = 0;
        int nC = 0;

        numRowOriginal = rowLowerOriginal.size();
        numColOriginal = colLowerOriginal.size();
        //arrays to keep track of indices
        vector<int> rIndex_(numRowOriginal, -1);
        vector<int> cIndex_(numColOriginal, -1);

        for (i=0;i<numRowOriginal;++i)
                if (flagRow[i]) {
                        for (j = ARstart[i]; j<ARstart.at(i+1); ++j)
                                if (flagCol[ARindex[j]])
                                        nz ++;
                        rIndex_[i] = nR;
                        nR++;
                        }

        for (i=0;i<numColOriginal;++i)
                if (flagCol[i]) {
                        cIndex_[i] = nC;
                        nC++;
                }


        //matrix
        vector<int>    Mstart(nC + 1, 0);
        vector<int>    Mindex(nz);
        vector<double> Mvalue(nz);

    vector<int> iwork(nC, 0);

    for (i = 0;i<numRowOriginal; ++i)
        if (flagRow[i])
            for (int k = ARstart[i]; k < ARstart.at(i+1);++k ) {
                j = ARindex[k];
                if (flagCol[j])
                                iwork[cIndex_[j]]++;
                        }
    for (i = 1; i <= nC; ++i)
        Mstart[i] = Mstart[i - 1] + iwork[i - 1];
   for (i = 0; i < numColOriginal; ++i)
        iwork[i] = Mstart[i];

   for (i = 0; i < numRowOriginal; ++i) {
        if (flagRow[i]) {
                        int iRow = rIndex_[i];
                    for (k = ARstart[i]; k < ARstart[i + 1];++k ) {
                        j = ARindex[k];
                        if (flagCol[j]) {
                                int iCol = cIndex_[j];
                                    int iPut = iwork[iCol]++;
                                    Mindex[iPut] = iRow;
                                    Mvalue[iPut] = ARvalue[k];
                                }
                    }
                }
    }

    vector<int>  bindex(nR);
    int countBasic=0;

    printf("To recover this test need to use col/row_status\n");
     for (int i=0; i< nonbasicFlag.size();++i) {
         if (nonbasicFlag[i] == 0)
                         countBasic++;
     }

     if (countBasic != nR)
         cout<<" Wrong count of basic variables: != numRow"<<endl;

     int c=0;
     for (int i=0; i< nonbasicFlag.size();++i) {
         if (nonbasicFlag[i] == 0) {
                        if (i < numColOriginal)
                                bindex[c] = cIndex_[i];
                        else
                                bindex[c] = nC + rIndex_[i - numColOriginal];
                        c++;
         }
    }

        factor.setup(nC, nR, &Mstart[0], &Mindex[0], &Mvalue[0],  &bindex[0]);
/ *	if (1) // for this check both A and M are the full matrix again
        {
                if (nC - numColOriginal != 0)
                        cout<<"columns\n";
                if (nR - numRowOriginal != 0)
                        cout<<"rows\n";
                for (int i=0; i< Mstart.size();++i)
                        if (Mstart[i] - Astart[i] != 0)
                                cout<<"Mstart "<<i<<"\n";
                for (int i=0; i< Mindex.size();++i)
                        if (Mindex[i] - Aindex[i] != 0)
                                cout<<"Mindex "<<i<<"\n";
                for (int i=0; i< Mvalue.size();++i)
                        if (Mvalue[i] - Avalue[i] != 0)
                                cout<<"Mvalue "<<i<<"\n";
                for (int i=0; i< bindex.size();++i)
                        if (nonbasicFlag[i] - nbffull[i] != 0)
                                cout<<"nbf "<<i<<"\n";
        } * /

        try {
        factor.build();
    } catch (runtime_error& error) {
        cout << error.what() << endl;
        cout << "Postsolve: could not factorize basis matrix." << endl;
        return 0;
    }
    cout << "Postsolve: basis matrix successfully factorized." << endl;

    return 1;
}*/

/***
 * lo and up refer to the place storing the current bounds on y_row
 *
 */
void Presolve::getBoundOnLByZj(int row, int j, double* lo, double* up,
                               double colLow, double colUpp) {
  double cost = colCostAtEl[j];  // valueColDual[j];
  double x = -cost;

  double sum = 0;
  for (int kk = Astart[j]; kk < Aend[j]; ++kk)
    if (flagRow[Aindex[kk]]) {
      sum = sum + Avalue[kk] * valueRowDual[Aindex[kk]];
    }
  x = x - sum;

  double aij = getaij(row, j);
  x = x / aij;

  if (fabs(colLow - colUpp) < tol)
    return;  // here there is no restriction on zj so no bound on y

  if ((valuePrimal[j] - colLow) > tol && (colUpp - valuePrimal[j]) > tol) {
    // set both bounds
    if (x < *up) *up = x;
    if (x > *lo) *lo = x;
  }

  else if ((valuePrimal[j] == colLow && aij < 0) ||
           (valuePrimal[j] == colUpp && aij > 0)) {
    if (x < *up) *up = x;
  } else if ((valuePrimal[j] == colLow && aij > 0) ||
             (valuePrimal[j] == colUpp && aij < 0)) {
    if (x > *lo) *lo = x;
  }
}

/**
 * returns z_col
 * z = A'y + c
 */
double Presolve::getColumnDualPost(int col) {
  int row;
  double z;
  double sum = 0;
  for (int cnt = Astart[col]; cnt < Aend[col]; cnt++)
    if (flagRow[Aindex[cnt]]) {
      row = Aindex[cnt];
      sum = sum + valueRowDual[row] * Avalue[cnt];
    }
  z = sum + colCostAtEl[col];
  return z;
}

/***
 * A'y + c = z
 *
 * returns y_row = -(A'y      +   c   - z )/a_rowcol
 *               (except row)  (at el)
 */
double Presolve::getRowDualPost(int row, int col) {
  double x = 0;

  for (int kk = Astart[col]; kk < Aend[col]; ++kk)
    if (flagRow[Aindex[kk]] && Aindex[kk] != row)
      x = x + Avalue[kk] * valueRowDual[Aindex[kk]];

  x = x + colCostAtEl[col] - valueColDual[col];

  double y = getaij(row, col);
  return -x / y;
}

string Presolve::getDualsForcingRow(int row, vector<int>& fRjs) {
  double z;
  stringstream ss;
  int j;

  double lo = -HIGHS_CONST_INF;
  double up = HIGHS_CONST_INF;
  int lo_col = -1;
  int up_col = -1;

  double cost, sum;

  for (size_t jj = 0; jj < fRjs.size(); ++jj) {
    j = fRjs[jj];

    pair<int, vector<double>> p = oldBounds.top();
    vector<double> v = get<1>(p);
    oldBounds.pop();
    double colLow = v[0];
    double colUpp = v[1];

    // calculate bound x imposed by zj
    double save_lo = lo;
    double save_up = up;
    getBoundOnLByZj(row, j, &lo, &up, colLow, colUpp);
    if (lo > save_lo) lo_col = j;
    if (up < save_up) up_col = j;
  }

  // calculate yi
  if (lo > up)
    cout << "PR: Error in postsolving forcing row " << row
         << " : inconsistent bounds for its dual value.\n";

  if (lo <= 0 && up >= 0) {
    valueRowDual[row] = 0;
    row_status[row] = HighsBasisStatus::BASIC;
  } else if (lo > 0) {
    // row is set to basic and column to non-basic but that should change
    row_status[row] = HighsBasisStatus::NONBASIC;
    col_status[lo_col] = HighsBasisStatus::BASIC;
    valueRowDual[row] = lo;
    valueColDual[lo_col] = 0;
    // valueColDual[lo_col] should be zero since it imposed the lower bound.
  } else if (up < 0) {
    // row is set to basic and column to non-basic but that should change
    row_status[row] = HighsBasisStatus::NONBASIC;
    col_status[up_col] = HighsBasisStatus::BASIC;
    valueRowDual[row] = up;
    valueColDual[up_col] = 0;
  }

  flagRow[row] = 1;

  for (size_t jj = 0; jj < fRjs.size(); ++jj) {
    j = fRjs[jj];
    if (lo > 0 && j == lo_col) continue;
    if (up < 0 && j == up_col) continue;

    col_status[j] = HighsBasisStatus::NONBASIC;

    cost = valueColDual[j];
    sum = 0;
    for (int k = Astart[j]; k < Aend[j]; ++k)
      if (flagRow[Aindex[k]]) {
        sum = sum + valueRowDual[Aindex[k]] * Avalue[k];
        // cout<<" row "<<Aindex[k]<<" dual
        // "<<valueRowDual[Aindex[k]]<<" a_"<<Aindex[k]<<"_"<<j<<"\n";
      }
    z = cost + sum;

    valueColDual[j] = z;

    if (iKKTcheck == 1) {
      ss << j;
      ss << " ";
      chk.addChange(2, 0, j, valuePrimal[j], valueColDual[j], cost);
    }
  }

  return ss.str();
}

void Presolve::getDualsSingletonRow(int row, int col) {
  pair<int, vector<double>> bnd = oldBounds.top();
  oldBounds.pop();

  valueRowDual[row] = 0;
  //   double cost = postValue.top();
  postValue.pop();
  double aij = getaij(row, col);
  double l = (get<1>(bnd))[0];
  double u = (get<1>(bnd))[1];
  double lrow = (get<1>(bnd))[2];
  double urow = (get<1>(bnd))[3];

  flagRow[row] = 1;

  HighsBasisStatus local_status;
  local_status = col_status[col];
  if (local_status != HighsBasisStatus::BASIC) {
    // x was not basic but is now
    // if x is strictly between original bounds or a_ij*x_j is at a bound.
    if (fabs(valuePrimal[col] - l) > tol && fabs(valuePrimal[col] - u) > tol) {
      if (report_postsolve) {
        printf("3.1 : Make column %3d basic and row %3d nonbasic\n", col, row);
      }
      col_status[col] = HighsBasisStatus::BASIC;
      row_status[row] = HighsBasisStatus::NONBASIC;  // Was LOWER
      valueColDual[col] = 0;
      valueRowDual[row] = getRowDualPost(row, col);
    } else {
      // column is at bound
      bool isRowAtLB = fabs(aij * valuePrimal[col] - lrow) < tol;
      bool isRowAtUB = fabs(aij * valuePrimal[col] - urow) < tol;

      double save_dual = valueColDual[col];
      valueColDual[col] = 0;
      double row_dual = getRowDualPost(row, col);

      if ((isRowAtLB && !isRowAtUB && row_dual > 0) ||
          (!isRowAtLB && isRowAtUB && row_dual < 0) ||
          (!isRowAtLB && !isRowAtUB)) {
        // make row basic
        row_status[row] = HighsBasisStatus::BASIC;
        valueRowDual[row] = 0;
        valueColDual[col] = save_dual;
      } else {
        // column is basic
        col_status[col] = HighsBasisStatus::BASIC;
        row_status[row] = HighsBasisStatus::NONBASIC;
        valueColDual[col] = 0;
        valueRowDual[row] = getRowDualPost(row, col);
      }
    }
  } else {
    // x is basic
    if (report_postsolve) {
      printf("3.3 : Make row %3d basic\n", row);
    }
    row_status[row] = HighsBasisStatus::BASIC;
    valueRowDual[row] = 0;
    // if the row dual is zero it does not contribute to the column dual.
  }

  if (iKKTcheck == 1) {
    chk.colDual[col] = valueColDual[col];
    chk.rowDual[row] = valueRowDual[row];
  }
}

void Presolve::getDualsDoubletonEquation(int row, int col) {
  // colDual already set. need valuePrimal from stack. maybe change rowDual
  // depending on bounds. old bounds kept in oldBounds. variables j,k : we
  // eliminated col(k)(c.col) and are left with changed bounds on j and no row.
  //                               y x

  pair<int, vector<double>> p = oldBounds.top();
  oldBounds.pop();
  vector<double> v = get<1>(p);
  int x = get<0>(p);
  double ubxNew = v[1];
  double lbxNew = v[0];
  double cxNew = v[2];
  p = oldBounds.top();
  oldBounds.pop();
  v = get<1>(p);
  double ubxOld = v[1];
  double lbxOld = v[0];
  double cxOld = v[2];
  p = oldBounds.top();
  oldBounds.pop();
  v = get<1>(p);
  double uby = v[1];
  double lby = v[0];
  double cy = v[2];

  int y = col;

  double b = postValue.top();
  postValue.pop();
  double aky = postValue.top();
  postValue.pop();
  double akx = postValue.top();
  postValue.pop();
  double valueX = valuePrimal[x];

  // primal value and objective shift
  valuePrimal[y] = (b - akx * valueX) / aky;
  objShift += -cxNew * valueX + cxOld * valueX + cy * valuePrimal[y];

  // column cost of x
  colCostAtEl[x] = cxOld;

  flagRow[row] = 1;

  HighsBasisStatus local_status;
  if (x < numColOriginal) {
    local_status = col_status[x];
  } else {
    local_status = row_status[x - numColOriginal];
  }
  if ((local_status != HighsBasisStatus::BASIC && valueX == ubxNew &&
       ubxNew < ubxOld) ||
      (local_status != HighsBasisStatus::BASIC && valueX == lbxNew &&
       lbxNew > lbxOld)) {
    if (x < numColOriginal) {
      col_status[x] = HighsBasisStatus::BASIC;
      // transfer dual of x to dual of row
      valueColDual[x] = 0;
      valueRowDual[row] = getRowDualPost(row, x);
      valueColDual[y] = getColumnDualPost(y);

      if (report_postsolve) printf("4.1 : Make column %3d basic\n", x);
    } else {
      row_status[x - numColOriginal] = HighsBasisStatus::BASIC;
      if (report_postsolve)
        printf("4.1 : Make row    %3d basic\n", x - numColOriginal);

      valueRowDual[row] = 0;
      valueColDual[x] = getColumnDualPost(x);
      valueColDual[y] = getColumnDualPost(y);
    }
  } else {
    // row becomes basic unless y is between bounds, in which case y is basic
    if (valuePrimal[y] - lby > tol && uby - valuePrimal[y] > tol) {
      if (y < numColOriginal) {
        col_status[y] = HighsBasisStatus::BASIC;
        if (report_postsolve) printf("4.2 : Make column %3d basic\n", y);

        valueColDual[y] = 0;
        valueRowDual[row] = getRowDualPost(row, y);
      } else {
        row_status[y - numColOriginal] = HighsBasisStatus::BASIC;
        if (report_postsolve)
          printf("4.2 : Make row    %3d basic\n", y - numColOriginal);

        valueRowDual[row] = 0;
        valueColDual[x] = getColumnDualPost(x);
        valueColDual[y] = getColumnDualPost(y);
      }
    } else if (fabs(valueX - ubxNew) < tol || fabs(valueX - lbxNew) < tol) {
      if (y < numColOriginal) {
        col_status[y] = HighsBasisStatus::BASIC;
        if (report_postsolve) printf("4.3 : Make column %3d basic\n", y);

        valueColDual[y] = 0;
        valueRowDual[row] = getRowDualPost(row, y);
      } else {
        row_status[y - numColOriginal] = HighsBasisStatus::BASIC;
        if (report_postsolve)
          printf("4.3 : Make row    %3d basic\n", y - numColOriginal);

        valueRowDual[row] = 0;
        valueColDual[x] = getColumnDualPost(x);
        valueColDual[y] = getColumnDualPost(y);
      }
    } else {
      if (report_postsolve) {
        printf("4.4 : Make row    %3d basic\n", row);
      }
      row_status[row] = HighsBasisStatus::BASIC;
      valueRowDual[row] = 0;
      valueColDual[x] = getColumnDualPost(x);
      valueColDual[y] = getColumnDualPost(y);
    }
  }
  if (iKKTcheck == 1) {
    chk.colDual[x] = valueColDual[x];
    chk.colDual[y] = valueColDual[y];
    chk.rowDual[row] = valueRowDual[row];
  }

  flagCol[y] = 1;
}

void Presolve::countRemovedRows(PresolveRule rule) {
  timer.increaseCount(true, rule);
}

void Presolve::countRemovedCols(PresolveRule rule) {
  timer.increaseCount(false, rule);
}

}  // namespace presolve
