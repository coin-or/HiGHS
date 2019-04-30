/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2018 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file util/HUtils.h
 * @brief Class-independent utilities for HiGHS
 * @author Julian Hall, Ivet Galabova, Qi Huangfu and Michael Feldmeier
 */
#ifndef UTIL_HIGHSUTILS_H_
#define UTIL_HIGHSUTILS_H_

#include <vector>

#include "HConfig.h"

/**
 * @brief Logical check of double being +Infinity
 */
bool highs_isInfinity(
		      double val //!< Value being tested against +Infinity
		      );
/**
 * @brief Analyse the values of a vector, assessing how many are in
 * each power of ten, and possibly analyse the distribution of
 * different values
 */
  void util_analyseVectorValues(
				const char* message,            //!< Message to be printed
				int vecDim,                     //!< Dimension of vector
				const std::vector<double>& vec, //!< Vector of values
				bool analyseValueList           //!< Possibly analyse the distribution of different values in the vector
				);

  void util_analyseMatrixSparsity(
				  const char* message,            //!< Message to be printed
				  int numCol,                     //!< Number of columns
				  int numRow,                     //!< Number of rows
				  const std::vector<int>& Astart, //!< Matrix column starts
				  const std::vector<int>& Aindex  //!< Matrix row indices
				  );
#endif // UTIL_HIGHSUTILS_H_
