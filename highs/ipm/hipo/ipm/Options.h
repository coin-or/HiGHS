#ifndef HIPO_OPTION_H
#define HIPO_OPTION_H

#include "Parameters.h"
#include "io/HighsIO.h"
#include "lp_data/HighsOptions.h"

namespace hipo {

enum OptionNla {
  kOptionNlaMin = 0,
  kOptionNlaAugmented = kOptionNlaMin,
  kOptionNlaNormEq,
  kOptionNlaChoose,
  kOptionNlaMax = kOptionNlaChoose,
  kOptionNlaDefault = kOptionNlaChoose
};

struct Options {
  // Solver options
  OptionNla nla = kOptionNlaDefault;
  std::string crossover = kHighsOffString;
  std::string parallel = kHighsChooseString;
  std::string parallel_type = kHipoBothString;
  std::string ordering = kHighsChooseString;

  // Ipm parameters
  Int max_iter = kMaxIterDefault;
  double feasibility_tol = kIpmTolDefault;
  double optimality_tol = kIpmTolDefault;
  double crossover_tol = kIpmTolDefault;
  bool refine_with_ipx = true;
  double time_limit = -1.0;
  Int block_size = 0;

  // Logging
  bool display = true;
  bool timeless_log = false;
  const HighsLogOptions* log_options = nullptr;
};

}  // namespace hipo

#endif