/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2022 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Leona Gottwald and Michael    */
/*    Feldmeier                                                          */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file io/FilereaderLp.cpp
 * @brief
 */

#include "io/FilereaderNag.h"

#include <cstdarg>

#define NAG_MAX_LINE_LENGTH 99

#include "lp_data/HighsLpUtils.h"

FilereaderRetcode FilereaderNag::readModelFromFile(const HighsOptions& options,
                                                  const std::string filename,
                                                  HighsModel& model) {
  return FilereaderRetcode::kNotImplemented;
}

void FilereaderNag::writeToFile(FILE* file, const char* format, ...) {
  va_list argptr;
  va_start(argptr, format);
  char stringbuffer[NAG_MAX_LINE_LENGTH + 1];
  HighsInt tokenlength = vsprintf(stringbuffer, format, argptr);
  if (this->linelength + tokenlength >= NAG_MAX_LINE_LENGTH) {
    fprintf(file, "\n");
    fprintf(file, "%s", stringbuffer);
    this->linelength = tokenlength;
  } else {
    fprintf(file, "%s", stringbuffer);
    this->linelength += tokenlength;
  }
}

void FilereaderNag::writeToFileLineend(FILE* file) {
  fprintf(file, "\n");
  this->linelength = 0;
}

HighsStatus FilereaderNag::writeModelToFile(const HighsOptions& options,
                                           const std::string filename,
                                           const HighsModel& model) {
  const HighsLp& lp = model.lp_;
  assert(lp.a_matrix_.isColwise());
  FILE* file = fopen(filename.c_str(), "w");

  // write start comment
  this->writeToFile(file, "nag_opt_qpconvex1_sparse_solve (HiGHS) Program Data");
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // write problem dimensions
  this->writeToFile(file, "Values of n and m");
  this->writeToFileLineend(file);
  this->writeToFile(file, "%d %d", model.lp_.num_col_, model.lp_.num_row_+1);
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // write problem dimensions
  this->writeToFile(file, "Values of nnz, iobj and ncolh");
  this->writeToFileLineend(file);
  this->writeToFile(file, "%d %d %d", model.lp_.a_matrix_.start_[model.lp_.num_col_] + model.lp_.num_col_, model.lp_.num_row_+1, model.lp_.num_col_);
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // write constraint matrix + linear objective
  this->writeToFile(file, "Matrix nonzeros: value, row index, column index");
  this->writeToFileLineend(file);
  for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    for (HighsInt index=model.lp_.a_matrix_.start_[col]; index < model.lp_.a_matrix_.start_[col+1]; index++) {
      this->writeToFile(file, "%lf %d %d", model.lp_.a_matrix_.value_[index], model.lp_.a_matrix_.index_[index]+1, col+1);
      this->writeToFileLineend(file);
    }
    this->writeToFile(file, "%lf %d %d", model.lp_.col_cost_[col], model.lp_.num_row_+1, col+1);
    this->writeToFileLineend(file);
  }
  this->writeToFileLineend(file);

  // write lower bounds
  this->writeToFile(file, "Lower bounds");
  this->writeToFileLineend(file);
  for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    if (model.lp_.col_lower_[col] < -10E25) {
      this->writeToFile(file, "-1e+25 ");
    } else {
      this->writeToFile(file, "%lf ", model.lp_.col_lower_[col]);
    }
  }
  for (HighsInt row=0; row<model.lp_.num_row_; row++) {
    if (model.lp_.row_lower_[row] < -1E25) {
      this->writeToFile(file, "-1e+25 ");
    } else {
      this->writeToFile(file, "%lf ", model.lp_.row_lower_[row]);
    }
  }
  this->writeToFile(file, "%s ", "-1e+25");
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // write upper bounds
  this->writeToFile(file, "Upper bounds");
  this->writeToFileLineend(file);
  for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    if (model.lp_.col_upper_[col] > 1E25) {
      this->writeToFile(file, "1e+25 ");
    } else {
      this->writeToFile(file, "%lf ", model.lp_.col_upper_[col]);
    }
  }
  for (HighsInt row=0; row<model.lp_.num_row_; row++) {
    if (model.lp_.row_upper_[row] > 1E25) {
      this->writeToFile(file, "1e+25 ");
    } else {
      this->writeToFile(file, "%lf ", model.lp_.row_upper_[row]);
    }
  }
  this->writeToFile(file, "%s ", "1e+25");
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // write column + row names
  this->writeToFile(file, "Column and row names");
  this->writeToFileLineend(file);
  for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    this->writeToFile(file, "'C %6d' ", col+1);
  }
  this->writeToFileLineend(file);
  for (HighsInt row=0; row<model.lp_.num_row_; row++) {
    this->writeToFile(file, "'R %6d' ", row+1);
  }
  this->writeToFile(file, "'OBJECTIV'");
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // initial estimate
  this->writeToFile(file, "Initial estimate of x");
  this->writeToFileLineend(file);
    for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    this->writeToFile(file, "%lf ", 0.0);
  }
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // hessian dimension
  this->writeToFile(file, "Number of hessian nonzeros");
  this->writeToFileLineend(file);
  this->writeToFile(file, "%d ", model.hessian_.numNz());
  this->writeToFileLineend(file);
  this->writeToFileLineend(file);

  // hessian
  this->writeToFile(file, "Hessian nonzeros: value, row index, col index (diagonal/lower triangle elements)");
  this->writeToFileLineend(file);
  for (HighsInt col=0; col<model.lp_.num_col_; col++) {
    for (HighsInt index=model.hessian_.start_[col]; index < model.hessian_.start_[col+1]; index++) {
      this->writeToFile(file, "%lf %d %d", model.hessian_.value_[index], model.hessian_.index_[index]+1, col+1);
      this->writeToFileLineend(file);
    }
  }
  this->writeToFileLineend(file);

  fclose(file);
  return HighsStatus::kOk;
}
