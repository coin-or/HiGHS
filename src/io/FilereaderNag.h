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

#ifndef IO_FILEREADER_NAG_H_
#define IO_FILEREADER_NAG_H_

#include <list>

#include "io/Filereader.h"
#include "io/HighsIO.h"

class FilereaderNag : public Filereader {
 public:
  FilereaderRetcode readModelFromFile(const HighsOptions& options,
                                      const std::string filename,
                                      HighsModel& model);

  HighsStatus writeModelToFile(const HighsOptions& options,
                               const std::string filename,
                               const HighsModel& model);

 private:
  // functions to write files
  HighsInt linelength;
  void writeToFile(FILE* file, const char* format, ...);
  void writeToFileLineend(FILE* file);
};

#endif
