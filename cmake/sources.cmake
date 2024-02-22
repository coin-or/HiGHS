set(cupdlp_sources
  src/pdlp/cupdlp/cupdlp_cs.c
  src/pdlp/cupdlp/cupdlp_linalg.c
  src/pdlp/cupdlp/cupdlp_proj.c
  src/pdlp/cupdlp/cupdlp_restart.c
  src/pdlp/cupdlp/cupdlp_scaling_cuda.c
  src/pdlp/cupdlp/cupdlp_solver.c
  src/pdlp/cupdlp/cupdlp_step.c
  src/pdlp/cupdlp/cupdlp_utils.c)

set(basiclu_sources
  src/ipm/basiclu/basiclu_factorize.c
  src/ipm/basiclu/basiclu_solve_dense.c
  src/ipm/basiclu/lu_build_factors.c
  src/ipm/basiclu/lu_factorize_bump.c
  src/ipm/basiclu/lu_initialize.c
  src/ipm/basiclu/lu_markowitz.c
  src/ipm/basiclu/lu_setup_bump.c
  src/ipm/basiclu/lu_solve_sparse.c
  src/ipm/basiclu/basiclu_get_factors.c
  src/ipm/basiclu/basiclu_solve_for_update.c
  src/ipm/basiclu/lu_condest.c
  src/ipm/basiclu/lu_file.c
  src/ipm/basiclu/lu_internal.c
  src/ipm/basiclu/lu_matrix_norm.c
  src/ipm/basiclu/lu_singletons.c
  src/ipm/basiclu/lu_solve_symbolic.c
  src/ipm/basiclu/lu_update.c
  src/ipm/basiclu/basiclu_initialize.c
  src/ipm/basiclu/basiclu_solve_sparse.c
  src/ipm/basiclu/lu_pivot.c
  src/ipm/basiclu/lu_solve_dense.c
  src/ipm/basiclu/lu_solve_triangular.c
  src/ipm/basiclu/basiclu_object.c
  src/ipm/basiclu/basiclu_update.c
  src/ipm/basiclu/lu_dfs.c
  src/ipm/basiclu/lu_garbage_perm.c
  src/ipm/basiclu/lu_residual_test.c
  src/ipm/basiclu/lu_solve_for_update.c)

set(ipx_sources
  src/ipm/ipx/basiclu_kernel.cc
  src/ipm/ipx/basiclu_wrapper.cc
  src/ipm/ipx/basis.cc
  src/ipm/ipx/conjugate_residuals.cc
  src/ipm/ipx/control.cc
  src/ipm/ipx/crossover.cc
  src/ipm/ipx/diagonal_precond.cc
  src/ipm/ipx/forrest_tomlin.cc
  src/ipm/ipx/guess_basis.cc
  src/ipm/ipx/indexed_vector.cc
  src/ipm/ipx/info.cc
  src/ipm/ipx/ipm.cc
  src/ipm/ipx/ipx_c.cc
  src/ipm/ipx/iterate.cc
  src/ipm/ipx/kkt_solver.cc
  src/ipm/ipx/kkt_solver_basis.cc
  src/ipm/ipx/kkt_solver_diag.cc
  src/ipm/ipx/linear_operator.cc
  src/ipm/ipx/lp_solver.cc
  src/ipm/ipx/lu_factorization.cc
  src/ipm/ipx/lu_update.cc
  src/ipm/ipx/maxvolume.cc
  src/ipm/ipx/model.cc
  src/ipm/ipx/normal_matrix.cc
  src/ipm/ipx/sparse_matrix.cc
  src/ipm/ipx/sparse_utils.cc
  src/ipm/ipx/splitted_normal_matrix.cc
  src/ipm/ipx/starting_basis.cc
  src/ipm/ipx/symbolic_invert.cc
  src/ipm/ipx/timer.cc
  src/ipm/ipx/utils.cc
  src/ipm/IpxWrapper.cpp)

set(highs_sources
    extern/filereaderlp/reader.cpp
    src/io/Filereader.cpp
    src/io/FilereaderLp.cpp
    src/io/FilereaderEms.cpp
    src/io/FilereaderMps.cpp
    src/io/HighsIO.cpp
    src/io/HMPSIO.cpp
    src/io/HMpsFF.cpp
    src/io/LoadOptions.cpp
    src/lp_data/Highs.cpp
    src/lp_data/HighsCallback.cpp
    src/lp_data/HighsDebug.cpp
    src/lp_data/HighsDeprecated.cpp
    src/lp_data/HighsInfo.cpp
    src/lp_data/HighsInfoDebug.cpp
    src/lp_data/HighsInterface.cpp
    src/lp_data/HighsLp.cpp
    src/lp_data/HighsLpUtils.cpp
    src/lp_data/HighsModelUtils.cpp
    src/lp_data/HighsRanging.cpp
    src/lp_data/HighsSolution.cpp
    src/lp_data/HighsSolutionDebug.cpp
    src/lp_data/HighsSolve.cpp
    src/lp_data/HighsStatus.cpp
    src/lp_data/HighsOptions.cpp
    src/parallel/HighsTaskExecutor.cpp
    src/pdlp/CupdlpWrapper.cpp
    src/presolve/ICrash.cpp
    src/presolve/ICrashUtil.cpp
    src/presolve/ICrashX.cpp
    src/mip/HighsMipSolver.cpp
    src/mip/HighsMipSolverData.cpp
    src/mip/HighsDomain.cpp
    src/mip/HighsDynamicRowMatrix.cpp
    src/mip/HighsLpRelaxation.cpp
    src/mip/HighsSeparation.cpp
    src/mip/HighsSeparator.cpp
    src/mip/HighsTableauSeparator.cpp
    src/mip/HighsModkSeparator.cpp
    src/mip/HighsPathSeparator.cpp
    src/mip/HighsCutGeneration.cpp
    src/mip/HighsSearch.cpp
    src/mip/HighsConflictPool.cpp
    src/mip/HighsCutPool.cpp
    src/mip/HighsCliqueTable.cpp
    src/mip/HighsGFkSolve.cpp
    src/mip/HighsTransformedLp.cpp
    src/mip/HighsLpAggregator.cpp
    src/mip/HighsDebugSol.cpp
    src/mip/HighsImplications.cpp
    src/mip/HighsPrimalHeuristics.cpp
    src/mip/HighsPseudocost.cpp
    src/mip/HighsNodeQueue.cpp
    src/mip/HighsObjectiveFunction.cpp
    src/mip/HighsRedcostFixing.cpp
    src/model/HighsHessian.cpp
    src/model/HighsHessianUtils.cpp
    src/model/HighsModel.cpp
    src/parallel/HighsTaskExecutor.cpp
    src/presolve/ICrashX.cpp
    src/presolve/HighsPostsolveStack.cpp
    src/presolve/HighsSymmetry.cpp
    src/presolve/HPresolve.cpp
    src/presolve/HPresolveAnalysis.cpp
    src/presolve/PresolveComponent.cpp
    src/qpsolver/a_asm.cpp
    src/qpsolver/a_quass.cpp
    src/qpsolver/basis.cpp
    src/qpsolver/quass.cpp
    src/qpsolver/ratiotest.cpp
    src/qpsolver/scaling.cpp
    src/qpsolver/perturbation.cpp
    src/simplex/HEkk.cpp
    src/simplex/HEkkControl.cpp
    src/simplex/HEkkDebug.cpp
    src/simplex/HEkkPrimal.cpp
    src/simplex/HEkkDual.cpp
    src/simplex/HEkkDualRHS.cpp
    src/simplex/HEkkDualRow.cpp
    src/simplex/HEkkDualMulti.cpp
    src/simplex/HEkkInterface.cpp
    src/simplex/HighsSimplexAnalysis.cpp
    src/simplex/HSimplex.cpp
    src/simplex/HSimplexDebug.cpp
    src/simplex/HSimplexNla.cpp
    src/simplex/HSimplexNlaDebug.cpp
    src/simplex/HSimplexNlaFreeze.cpp
    src/simplex/HSimplexNlaProductForm.cpp
    src/simplex/HSimplexReport.cpp
    src/test/KktCh2.cpp
    src/test/DevKkt.cpp
    src/util/HFactor.cpp
    src/util/HFactorDebug.cpp
    src/util/HFactorExtend.cpp
    src/util/HFactorRefactor.cpp
    src/util/HFactorUtils.cpp
    src/util/HighsHash.cpp
    src/util/HighsLinearSumBounds.cpp
    src/util/HighsMatrixPic.cpp
    src/util/HighsMatrixUtils.cpp
    src/util/HighsSort.cpp
    src/util/HighsSparseMatrix.cpp
    src/util/HighsUtils.cpp
    src/util/HSet.cpp
    src/util/HVectorBase.cpp
    src/util/stringutil.cpp
    src/interfaces/highs_c_api.cpp)


set(headers_fast_build_
    ../extern/filereaderlp/builder.hpp
    ../extern/filereaderlp/model.hpp
    ../extern/filereaderlp/reader.hpp
    io/Filereader.h
    io/FilereaderLp.h
    io/FilereaderEms.h
    io/FilereaderMps.h
    io/HMpsFF.h
    io/HMPSIO.h
    io/HighsIO.h
    io/LoadOptions.h
    lp_data/HConst.h
    lp_data/HStruct.h
    lp_data/HighsAnalysis.h
    lp_data/HighsCallback.h
    lp_data/HighsCallbackStruct.h
    lp_data/HighsDebug.h
    lp_data/HighsInfo.h
    lp_data/HighsInfoDebug.h
    lp_data/HighsLp.h
    lp_data/HighsLpSolverObject.h
    lp_data/HighsLpUtils.h
    lp_data/HighsModelUtils.h
    lp_data/HighsOptions.h
    lp_data/HighsRanging.h
    lp_data/HighsRuntimeOptions.h
    lp_data/HighsSolution.h
    lp_data/HighsSolutionDebug.h
    lp_data/HighsSolve.h
    lp_data/HighsStatus.h
    src/mip/HighsCliqueTable.h
    src/mip/HighsCutGeneration.h
    src/mip/HighsConflictPool.h
    src/mip/HighsCutPool.h
    src/mip/HighsDebugSol.h
    src/mip/HighsDomainChange.h
    src/mip/HighsDomain.h
    src/mip/HighsDynamicRowMatrix.h
    src/mip/HighsGFkSolve.h
    src/mip/HighsImplications.h
    src/mip/HighsLpAggregator.h
    src/mip/HighsLpRelaxation.h
    src/mip/HighsMipSolverData.h
    src/mip/HighsMipSolver.h
    src/mip/HighsModkSeparator.h
    src/mip/HighsNodeQueue.h
    src/mip/HighsObjectiveFunction.h
    src/mip/HighsPathSeparator.h
    src/mip/HighsPrimalHeuristics.h
    src/mip/HighsPseudocost.h
    src/mip/HighsRedcostFixing.h
    src/mip/HighsSearch.h
    src/mip/HighsSeparation.h
    src/mip/HighsSeparator.h
    src/mip/HighsTableauSeparator.h
    src/mip/HighsTransformedLp.h
    src/model/HighsHessian.h
    src/model/HighsHessianUtils.h
    src/model/HighsModel.h
    src/parallel/HighsBinarySemaphore.h
    src/parallel/HighsCacheAlign.h
    src/parallel/HighsCombinable.h
    src/parallel/HighsMutex.h
    src/parallel/HighsParallel.h
    src/parallel/HighsRaceTimer.h
    src/parallel/HighsSchedulerConstants.h
    src/parallel/HighsSpinMutex.h
    src/parallel/HighsSplitDeque.h
    src/parallel/HighsTaskExecutor.h
    src/parallel/HighsTask.h
    src/qpsolver/a_asm.hpp
    src/qpsolver/a_quass.hpp
    src/qpsolver/quass.hpp
    src/qpsolver/vector.hpp
    src/qpsolver/scaling.hpp
    src/qpsolver/perturbation.hpp
    src/simplex/HApp.h
    src/simplex/HEkk.h
    src/simplex/HEkkDual.h
    src/simplex/HEkkDualRHS.h
    src/simplex/HEkkDualRow.h
    src/simplex/HEkkPrimal.h
    src/simplex/HighsSimplexAnalysis.h
    src/simplex/HSimplex.h
    src/simplex/HSimplexReport.h
    src/simplex/HSimplexDebug.h
    src/simplex/HSimplexNla.h
    src/simplex/SimplexConst.h
    src/simplex/SimplexStruct.h
    src/simplex/SimplexTimer.h
    src/presolve/ICrash.h
    src/presolve/ICrashUtil.h
    src/presolve/ICrashX.h
    src/presolve/HighsPostsolveStack.h
    src/presolve/HighsSymmetry.h
    src/presolve/HPresolve.h
    src/presolve/HPresolveAnalysis.h
    src/presolve/PresolveComponent.h
    src/test/DevKkt.h
    src/test/KktCh2.h
    src/util/FactorTimer.h
    src/util/HFactor.h
    src/util/HFactorConst.h
    src/util/HFactorDebug.h
    src/util/HighsCDouble.h
    src/util/HighsComponent.h
    src/util/HighsDataStack.h
    src/util/HighsDisjointSets.h
    src/util/HighsHash.h
    src/util/HighsHashTree.h
    src/util/HighsInt.h
    src/util/HighsIntegers.h
    src/util/HighsLinearSumBounds.h
    src/util/HighsMatrixPic.h
    src/util/HighsMatrixSlice.h
    src/util/HighsMatrixUtils.h
    src/util/HighsRandom.h
    src/util/HighsRbTree.h
    src/util/HighsSort.h
    src/util/HighsSparseMatrix.h
    src/util/HighsSparseVectorSum.h
    src/util/HighsSplay.h
    src/util/HighsTimer.h
    src/util/HighsUtils.h
    src/util/HSet.h
    src/util/HVector.h
    src/util/HVectorBase.h
    src/util/stringutil.h
    src/Highs.h
    src/interfaces/highs_c_api.h
  )

#   set(headers_fast_build_ ${headers_fast_build_} ipm/IpxWrapper.h ${basiclu_headers}
#     ${ipx_headers})

# todo: see which headers you need 

  # set_target_properties(highs PROPERTIES PUBLIC_HEADER "src/Highs.h;src/lp_data/HighsLp.h;src/lp_data/HighsLpSolverObject.h")

  # install the header files of highs
#   foreach(file ${headers_fast_build_})
#     get_filename_component(dir ${file} DIRECTORY)

#     if(NOT dir STREQUAL "")
#       string(REPLACE ../extern/ "" dir ${dir})
#     endif()

#     install(FILES ${file} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs/${dir})
#   endforeach()
#   install(FILES ${HIGHS_BINARY_DIR}/HConfig.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/highs)

  set(include_dirs
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/interfaces
    ${CMAKE_SOURCE_DIR}/src/io
    ${CMAKE_SOURCE_DIR}/src/ipm
    ${CMAKE_SOURCE_DIR}/src/ipm/ipx
    ${CMAKE_SOURCE_DIR}/src/ipm/basiclu
    ${CMAKE_SOURCE_DIR}/src/lp_data
    ${CMAKE_SOURCE_DIR}/src/mip
    ${CMAKE_SOURCE_DIR}/src/model
    ${CMAKE_SOURCE_DIR}/src/parallel
    ${CMAKE_SOURCE_DIR}/src/pdlp
    ${CMAKE_SOURCE_DIR}/src/pdlp/cupdlp
    ${CMAKE_SOURCE_DIR}/src/presolve
    ${CMAKE_SOURCE_DIR}/src/qpsolver
    ${CMAKE_SOURCE_DIR}/src/simplex
    ${CMAKE_SOURCE_DIR}/src/util
    ${CMAKE_SOURCE_DIR}/src/test
    ${CMAKE_SOURCE_DIR}/extern
    ${CMAKE_SOURCE_DIR}/extern/filereader
    ${CMAKE_SOURCE_DIR}/extern/pdqsort
    $<BUILD_INTERFACE:${HIGHS_BINARY_DIR}>
    
  )
    
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    # $<BUILD_INTERFACE:${HIGHS_BINARY_DIR}>
    # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/highs>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interfaces>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/io>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm/ipx>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ipm/basiclu>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lp_data>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mip>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/model>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/parallel>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/presolve>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/qpsolver>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simplex>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/util>
    # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    # $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/>
    # $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/filereader>
    # $<BUILD_INTERFACE:${HIGHS_SOURCE_DIR}/extern/pdqsort>
  