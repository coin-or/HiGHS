# Install script for directory: C:/Users/galab/code/HiGHS/highs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/galab/installs/highs-static-nohipo")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/filereaderlp/builder.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/filereaderlp/def.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/filereaderlp/model.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/filereaderlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/filereaderlp/reader.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdqsort" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/pdqsort/pdqsort.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/zstr" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/zstr/strict_fstream.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/zstr" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/../extern/zstr/zstr.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/interfaces" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/interfaces/highs_c_api.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/Filereader.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/FilereaderEms.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/FilereaderLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/FilereaderMps.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/HighsIO.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/HMpsFF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/HMPSIO.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/io" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/io/LoadOptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/IpxSolution.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/IpxWrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsCallback.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsCallbackStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsIis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsInfo.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsInfoDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsLpSolverObject.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsLpUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsModelUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsOptions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsRanging.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsSolution.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsSolutionDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsSolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HighsStatus.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/lp_data" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/lp_data/HStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/feasibilityjump.hh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsCliqueTable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsConflictPool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsCutGeneration.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsCutPool.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsDebugSol.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsDomain.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsDomainChange.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsDynamicRowMatrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsGFkSolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsImplications.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsLpAggregator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsLpRelaxation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsMipAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsMipSolver.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsMipSolverData.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsModkSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsNodeQueue.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsObjectiveFunction.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsPathSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsPrimalHeuristics.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsPseudocost.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsRedcostFixing.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsSearch.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsSeparation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsTableauSeparator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/HighsTransformedLp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/mip" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/mip/MipTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/model/HighsHessian.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/model/HighsHessianUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/model" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/model/HighsModel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsBinarySemaphore.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsCacheAlign.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsCombinable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsMutex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsParallel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsRaceTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsSchedulerConstants.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsSpinMutex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsSplitDeque.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsTask.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/parallel" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/parallel/HighsTaskExecutor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/CupdlpWrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/HighsPostsolveStack.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/HighsSymmetry.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/HPresolve.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/HPresolveAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/ICrash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/ICrashUtil.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/ICrashX.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/presolve" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/presolve/PresolveComponent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/a_asm.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/a_quass.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/basis.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/crashsolution.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/dantzigpricing.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/devexpricing.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/eventhandler.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/factor.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/feasibility_bounded.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/feasibility_highs.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/gradient.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/instance.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/matrix.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/perturbation.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/pricing.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/qpconst.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/qpvector.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/quass.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/ratiotest.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/runtime.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/scaling.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/settings.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/snippets.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/statistics.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/qpsolver" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/qpsolver/steepestedgepricing.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HApp.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HEkk.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HEkkDual.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HEkkDualRHS.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HEkkDualRow.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HEkkPrimal.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HighsSimplexAnalysis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HSimplex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HSimplexDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HSimplexNla.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/HSimplexReport.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/SimplexConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/SimplexStruct.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/simplex" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/simplex/SimplexTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/test_kkt" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/test_kkt/DevKkt.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/test_kkt" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/test_kkt/KktCh2.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/FactorTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HFactor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HFactorConst.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HFactorDebug.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsCDouble.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsComponent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsDataStack.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsDisjointSets.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsHashTree.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsInt.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsIntegers.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsLinearSumBounds.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsMatrixPic.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsMatrixSlice.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsMatrixUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsMemoryAllocation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsRandom.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsRbTree.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsSort.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsSparseMatrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsSparseVectorSum.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsSplay.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsTimer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HighsUtils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HSet.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HVector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/HVectorBase.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/util" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/util/stringutil.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/Highs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_cs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_defs.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_linalg.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_proj.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_restart.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_scaling.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_solver.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_step.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/pdlp/cupdlp" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/pdlp/cupdlp/cupdlp_utils.c")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/basiclu_kernel.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/basiclu_wrapper.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/basis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/conjugate_residuals.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/control.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/crossover.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/diagonal_precond.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/forrest_tomlin.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/guess_basis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/indexed_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/info.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipm.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_c.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_config.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_info.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_internal.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_parameters.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/ipx_status.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/iterate.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/kkt_solver_basis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/kkt_solver_diag.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/kkt_solver.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/linear_operator.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/lp_solver.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/lu_factorization.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/lu_update.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/maxvolume.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/model.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/multistream.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/normal_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/power_method.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/sparse_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/sparse_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/splitted_normal_matrix.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/starting_basis.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/symbolic_invert.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/timer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/ipx" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/ipx/utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_factorize.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_get_factors.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_initialize.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_factorize.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_free.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_get_factors.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_initialize.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_solve_dense.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_solve_for_update.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_solve_sparse.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_obj_update.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_object.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_solve_dense.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_solve_for_update.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_solve_sparse.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu_update.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/basiclu.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/lu_def.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/lu_file.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/lu_internal.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs/ipm/basiclu" TYPE FILE FILES "C:/Users/galab/code/HiGHS/highs/ipm/basiclu/lu_list.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cli;libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/highs" TYPE FILE FILES "C:/Users/galab/code/HiGHS/build-static-nohipo/HConfig.h")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/galab/code/HiGHS/build-static-nohipo/highs/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
