#ifndef LP_DATA_HIGHS_STATUS_H_
#define LP_DATA_HIGHS_STATUS_H_

#include <string>

// HiGHS status
enum class HighsStatus {
  OK,
    //  Info,
  Warning,
    /*
  NotImplemented,
  Init,
  LpError,
  OptionsError,
  PresolveError,
  SolutionError,
  PostsolveError,
  LpEmpty,
  ReachedDualObjectiveUpperBound,
  Unbounded,
  Infeasible,
  PrimalFeasible,
  DualFeasible,
  Optimal,
  Timeout,
  ReachedIterationLimit,
  NumericalDifficulties
    */
  Error
};

// Report a HighsStatus.
void HighsStatusReport(const char* message, HighsStatus status);

// Return a string representation of HighsStatus.
std::string HighsStatusToString(HighsStatus status);

// Return the maximum of two HighsStatus
HighsStatus worseStatus(HighsStatus status0, HighsStatus status1);
#endif
