# [Solvers](@id solvers)

## Introduction

HiGHS has implementations of the three main solution techniques for LP
(simplex, interior point and primal-dual hybrid gradient), and two
solution techniques for QP (active set and interior point). HiGHS will
choose the most appropriate technique for a given problem, but this
can be over-ridden by setting the option [__solver__](@ref
option-solver), with a discussion of its interpretation given
[below](@ref solver-option). HiGHS has just one MIP solver, and the
options to define which LP solver is used to solve the sub-problems
that it generates are discussed [below](@ref solver-option).

## LP

#### Simplex

HiGHS has efficient implementations of both the primal and dual
simplex methods, although the dual simplex solver is likely to be
faster and is more robust, so is used by default. The novel features
of the dual simplex solver are described in

_Parallelizing the dual revised simplex method_, Q. Huangfu and
J. A. J. Hall, Mathematical Programming Computation, 10 (1), 119-142,
2018 [DOI:
10.1007/s12532-017-0130-5](https://link.springer.com/article/10.1007/s12532-017-0130-5).

* Setting the option [__solver__](@ref option-solver) to "simplex" forces the simplex solver to be used
* The option [__simplex\_strategy__](@ref option-simplex_strategy)
  determines whether the primal solver or one of the parallel solvers is
  to be used.

#### Interior point

HiGHS has two interior point (IPM) solvers:

* IPX is based on the preconditioned conjugate gradient method, as discussed in

  _Implementation of an interior point method with basis
  preconditioning_, Mathematical Programming Computation, 12, 603-635, 2020. [DOI:
  10.1007/s12532-020-00181-8](https://link.springer.com/article/10.1007/s12532-020-00181-8).

  This solver is serial.

* HiPO is based on a direct factorisation, as discussed in 

  _A factorisation-based regularised interior point method using the augmented system_, F. Zanetti and J. Gondzio, 2025, 
  [available on arxiv](https://arxiv.org/abs/2508.04370)

  This solver is parallel.

  The [hipo\_system](@ref option-hipo-system) option can be used to select the approach to use when solving the Newton systems 
  within the interior point solver: select "augmented" to force the solver to use the augmented system, "normaleq" for normal 
  equations, or "choose" to leave the choice to the solver.

  The option [hipo\_ordering](@ref option-hipo-ordering) can be used to select the fill-reducing heuristic to use during the factorisation:
  * Nested dissection, obtained setting the option [hipo\_ordering](@ref option-hipo-ordering) to "metis".
  * Approximate mininum degree, obtained setting the option [hipo\_ordering](@ref option-hipo-ordering) to "amd".
  * Reverse Cuthill-McKee, obtained setting the option [hipo\_ordering](@ref option-hipo-ordering) to "rcm".

#### Primal-dual hybrid gradient method

HiGHS has a primal-dual hybrid gradient implementation for LP (PDLP)
that can be run on an NVIDIA [GPU](@ref gpu) if CUDA is installed. On
a CPU, it is unlikely to be competitive with the HiGHS interior point
or simplex solvers.

## QP

The HiGHS solver for convex QP problems uses an established primal
active set method. The new interior point solver HiPO will soon be able to
solve convex QP problems.

## [MIP](@id mip-solver)

The HiGHS MIP solver uses established branch-and-cut techniques. It is
largely single-threaded, although implementing a multi-threaded tree
search is work in progress.

## [Solver option](@id solver-option)

The [__solver__](@ref option-solver) 


  Setting the option [__solver__](@ref option-solver) to "ipx" forces the IPX solver to be used
  Setting the option [__solver__](@ref option-solver) to "hipo" forces the HiPO solver to be used.
Setting the option [__solver__](@ref option-solver) to "ipm" selects the HiPO solver, if the build supports it, otherwise it selects the IPX solver.




Setting the option [__solver__](@ref option-solver) to "pdlp" forces the PDLP solver to be used

