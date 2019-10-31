#include <cstdio>

#include "Highs.h"
#include "util/HighsUtils.h"
#include "io/LoadProblem.h"
#include "test/TestLp.h"
#include "TestInstanceLoad.h"
#include "catch.hpp"

#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#define NOGDI
#include <windows.h>
#else

#endif

const double kOptimalQap04 = 32;

// todo: write test with generateTestLpMini()
// same example as in notebook for breakpoints
// but as an equality constrained lp
// use both for unit test and debugging so define in
// test/ as mini-ff and mini-ff-i

// No commas in test case name.
TEST_CASE("ff-mini", "[highs_presolve]") {
  HighsLp lp = generateTestLpMini();

  Highs highs;
  HighsOptions options;
  options.find_feasibility = "on";
  highs.options_ = options;
  HighsStatus init_status = highs.initializeLp(lp);
  REQUIRE(init_status == HighsStatus::OK);

  HighsStatus run_status = highs.run();
  REQUIRE(run_status == HighsStatus::OK);

  const HighsSolution& x = highs.getSolution();

  // assert objective is close enough to the optimal one
  // c'x
  double ctx = highs.getObjectiveValue();

  // todo: add value of difference with mini lp oprimal objective
  double difference = std::fabs(0);
  REQUIRE(difference < 1e01);

  // assert residual is below threshold
  // r
  std::vector<double> residual;
  HighsSolution solution = highs.getSolution();

  if (solution.col_value.size() == 0) return;

  HighsStatus result = calculateResidual(highs.getLp(), solution, residual);
  REQUIRE(result == HighsStatus::OK);

  double r = getNorm2(residual);
  REQUIRE(r < 1e-08);
}

// No commas in test case name.
TEST_CASE("ff-qap04", "[highs_presolve]") {
  HighsOptions options;
  std::string dir = GetCurrentWorkingDir();

  std::cout << dir << std::endl;

  // For debugging use the latter.
  options.model_file = dir + "/../../check/instances/qap04.mps";
  //options.model_file = dir + "/check/instances/qap04.mps";

  HighsLp lp;
  HighsStatus read_status = loadLpFromFile(options, lp);
  REQUIRE(read_status == HighsStatus::OK);

  Highs highs;
  options.find_feasibility = "on";
  options.feasibility_initial_weight = 10;

  highs.options_ = options;
  HighsStatus init_status = highs.initializeLp(lp);
  REQUIRE(init_status == HighsStatus::OK);

  HighsStatus run_status = highs.run();
  REQUIRE(run_status == HighsStatus::OK);

  const HighsSolution& x = highs.getSolution();

  // assert objective is close enough to the optimal one
  // c'x
  double ctx = highs.getObjectiveValue();
  double difference = std::fabs(kOptimalQap04 - ctx);
  REQUIRE(difference < 0.18);

  // assert residual is below threshold
  // r
  std::vector<double> residual;
  HighsSolution solution = highs.getSolution();
  HighsStatus result = calculateResidual(highs.getLp(), solution, residual);
  REQUIRE(result == HighsStatus::OK);

  double r = getNorm2(residual);
  REQUIRE(r < 1e-08);
}