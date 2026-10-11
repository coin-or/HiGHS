/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsMiqp.cpp
 * @brief
 */
#include "lp_data/HighsMiqp.h"

#include "Highs.h"

HighsStatus Highs::solveMiqp() {
  HighsStatus status = HighsStatus::kOk;
  // Currently trying to solve a MIQP yields an error return
  status = this->optimizeModel();
  assert(status == HighsStatus::kError);

  // A handy const reference to the incumbent HighsLp
  HighsLp& lp = this->model_.lp_;
  // Extract the original number of columns and rows in the incumbent
  // HighsLp
  HighsInt num_col = lp.num_col_;
  HighsInt num_row = lp.num_row_;
  //
  // Note that HighsInt is the HiGHS integer: there is a build flag
  // (HIGHSINT64) that enables HiGHS to use 64-bit integers. Hence
  // printing of a HighsInt requires a cast to int if using %d,
  // otherise the PRId64d value can be used
  //
  // Illustrate the use of highsLogUser to perform logging: no other
  // method of permanent logging must be used
  highsLogUser(
      options_.log_options, HighsLogType::kInfo,
      "Highs::solveMiqp Solving an MIQP with %d columns and %" HIGHSINT_FORMAT
      " rows\n",
      int(num_col), num_row);

  // Move the Hessian into a local instance
  HighsHessian hessian = std::move(this->model_.hessian_);
  // Study the HighsHessian class (in highs/model/HighsHessian.h)!

  // Log information about the Hessian: also gives an illustration of
  // the Hessian properties
  assert(hessian.dim() == num_col);
  logHessian(options_.log_options, hessian, lp.integrality_);

  // Clear the Hessian in the incumbent HighsModel so that it is just
  // a MIP
  this->model_.hessian_.clear();

  // Linearising the quadratic terms in the Hessian requires rows and
  // (possibly) columns to be added to the incumbent HighsLp. The
  // following code (which only works for problems with at least 3
  // columns and rows) illustrates the process
  const bool illustration = true;
  if (illustration && lp.num_col_ >= 3 && lp.num_row_ >= 3) {
    highsLogUser(options_.log_options, HighsLogType::kInfo,
                 "One column and one row added to the problem as illustration "
                 "of process\n");

    // Lambda to find the maximum value in an std::vector<HighsInt>
    auto maxIndex = [&](const std::vector<HighsInt>& index) {
      HighsInt max_index = 0;
      for (HighsInt k = 0; k < int(index.size()); k++)
        max_index = std::max(index[k], max_index);
      return max_index;
    };

    // Add a column and row to the problem
    std::vector<HighsInt> index = {0, 1, lp.num_row_};
    std::vector<double> value = {1, 1, 1};
    assert(index.size() == value.size());
    HighsInt num_new_index = index.size();

    // Row indices must not exceed lp.num_row_ - 1, and this call
    // illustrates what happens if they do!
    status = this->addCol(1.0, 0, kHighsInf, num_new_index, index.data(),
                          value.data());
    assert(status == HighsStatus::kError);
    // If addCol/addRow returns an error, the incumbent HighsLp is not
    // changed
    assert(lp.num_col_ == num_col);

    // Now give a legal value to index[2]
    index[2] = lp.num_row_ - 1;

    assert(maxIndex(index) < lp.num_row_);
    status = this->addCol(1.0, 0, kHighsInf, num_new_index, index.data(),
                          value.data());
    assert(status == HighsStatus::kOk);
    assert(lp.num_col_ == num_col + 1);

    // The new column is continuous (HighsVarType::kContinuous) by
    // default. This is how to make it an integer variable.
    status = this->changeColIntegrality(1, HighsVarType::kInteger);
    assert(status == HighsStatus::kOk);

    // Now add a row
    index[2] = lp.num_col_ - 1;
    assert(maxIndex(index) < lp.num_col_);
    status =
        this->addRow(0, kHighsInf, num_new_index, index.data(), value.data());
    assert(status == HighsStatus::kOk);
    assert(lp.num_row_ == num_row + 1);
  }

  bool linearisable = hessian.isDiagonal();
  if (linearisable) {
    double kMaxUpper = 1e3;
    HighsInt iCol;
    for (iCol = 0; iCol < hessian.dim(); iCol++) {
      if (hessian.value_[iCol] &&
          (lp.col_lower_[iCol] != 0 || lp.col_upper_[iCol] > kMaxUpper)) {
        linearisable = false;
        break;
      }
    }
    if (!linearisable)
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "Highs::solveMiqp Cannot linearise since variable %8d has "
                   "bounds [%11.4g, %11.4g] and Hessian diagonal %11.4g\n",
                   int(iCol), lp.col_lower_[iCol], lp.col_upper_[iCol],
                   hessian.value_[iCol]);
  }

  // Once the linearisations have been added, the MILP is solved by
  // calling Highs::optimizeModel()
  this->reportModelStats();
  const HighsStatus run_status = this->optimizeModel();

  // Save data from the solution of the MIP, as this is cleared when
  // columns and rows are deleted
  HighsModelStatus model_status = this->model_status_;
  std::vector<double> col_value = this->solution_.col_value;
  std::vector<double> row_value = this->solution_.row_value;
  double objective_function_value = this->info_.objective_function_value;

  this->writeSolution("", 1);

  // Delete the columns and rows that have been added
  status = this->deleteCols(num_col, lp.num_col_ - 1);
  assert(status == HighsStatus::kOk);

  status = this->deleteRows(num_row, lp.num_row_ - 1);
  assert(status == HighsStatus::kOk);

  // Restore the Hessian in the incumbent HighsModel
  this->model_.hessian_ = std::move(hessian);
  this->reportModelStats();

  // Check that the columns and rows added have been removed
  assert(lp.num_col_ == num_col);
  assert(lp.num_row_ == num_row);

  // trim the column and row values to the original model dimesions
  col_value.resize(lp.num_col_);
  row_value.resize(lp.num_row_);
  this->model_status_ = model_status;
  this->solution_.col_value = col_value;
  this->solution_.row_value = row_value;
  this->solution_.value_valid = true;
  this->info_.objective_function_value = objective_function_value;
  this->writeSolution("", 1);

  // Currently still have to ensure that trying to solve a MIQP yields
  // an error return
  return HighsStatus::kError;  // run_status;
}

// Method to log Hessian information, serving as an example of how to
// set up a method not in the Highs class, with its header defined in
// HighsMiqp.h
void logHessian(const HighsLogOptions log_options, const HighsHessian& hessian,
                const std::vector<HighsVarType>& integrality) {
  // The Hessians received here will be triangular, with the first
  // nonzero in each column being the diagonal entry (possibly with
  // zero value). [Yes, hessian.numNz() is confusing...]
  assert(hessian.format_ == HessianFormat::kTriangular);
  bool diagonal = true;
  HighsInt num_diagonal_nz = 0;
  HighsInt num_diagonal_nz_int = 0;
  for (HighsInt iCol = 0; iCol < hessian.dim(); iCol++) {
    HighsInt iEl = hessian.start_[iCol];
    HighsInt iRow = hessian.index_[iEl];
    // First entry in each column is the diagonal
    assert(iRow == iCol);
    // Count the nonzeros on the diagonal, and the number of them that
    // correspond to integer variables
    if (hessian.value_[iEl]) {
      num_diagonal_nz++;
      if (integrality[iCol] == HighsVarType::kInteger) num_diagonal_nz_int++;
    }
    // If there are any other entries in this column, they will be
    // strictly nonzero, so the matrix is not diagonal
    if (hessian.start_[iCol + 1] > iEl + 1) diagonal = false;
  }
  assert(diagonal == hessian.isDiagonal());
  highsLogUser(
      log_options, HighsLogType::kInfo,
      "Highs::solveMiqp Hessian has %d entr%s, and is %sdiagonal,"
      " with %d / %d nonzero diagonal entr%s, of which %d correspond%s to "
      "%sinteger variable%s\n",
      int(hessian.numNz()), highsIntToPlural(hessian.numNz(), true).c_str(),
      diagonal ? "" : "not ", int(num_diagonal_nz), int(hessian.dim()),
      highsIntToPlural(num_diagonal_nz, true).c_str(), int(num_diagonal_nz_int),
      highsIntToThirdPerson(num_diagonal_nz_int).c_str(),
      highsIntToIndefiniteArticle(num_diagonal_nz_int, true).c_str(),
      highsIntToPlural(num_diagonal_nz_int).c_str());
}
