## Code changes

The HiPO release exposed various issues flagged up via GitHub and email.
- Fix some overflows when computing statistics of analyse phase.
- Free memory used for normal equations, if augmented system is preferred.
- Fix bug in supernode amalgamation.
- Print the BLAS library used in the HiGHS header, so it is visible when using HiPO without logging.
- Add the ability to use AMD and RCM rather than Metis
- Use 64-bit integers
- Fixed the time limit

Following PR [#2623](https://github.com/ERGO-Code/HiGHS/pull/2623),
singleton column stuffing added to MIP presolve - see Gamrath et al.,
Progress in presolving for mixed integer
programming. Math. Prog. Comp. 7, 367–398 (2015).

Following user PR
[#2625](https://github.com/ERGO-Code/HiGHS/issues/2625), callback data
structs are named, allowing them to be forward-declared data types if
a callback needs to be declared in a public header

Following PR [#2626](https://github.com/ERGO-Code/HiGHS/pull/2626),
`IPX` is used by default when switching to IPM after simplex reaches
iteration limit in `HighsLpRelaxation::run`

Following user PR
[#2628](https://github.com/ERGO-Code/HiGHS/pull/2628), `#include
<functional>` has been added in `highs/mip/HighsGFkSolve.h` and
`highs/mip/HighsNodeQueue.h` to avoid compilation failures for some
compilers.

Prompted by [#2633](https://github.com/ERGO-Code/HiGHS/issues/2633),
the constraint matrix is passed by reference (rather than value) for
each constraint when writing a LP file.

Following PR [#2639](https://github.com/ERGO-Code/HiGHS/pull/2639),
the dominated columns reduction is speeded up for models with many
columns.

Prompted by [#2643](https://github.com/ERGO-Code/HiGHS/issues/2643),
primal simplex is avoided when the unscaled LP problem has primal
infeasibilities but is dual feasible.

Following PR [#2644](https://github.com/ERGO-Code/HiGHS/pull/2644),
the irreducible infeasibility system detection facility has been
refactored and is much more robust. Rather than have
`HighsOption::iis_strategy` be one from an enum of "strategy
scenarios" it is now a bit map

- 0 => "light strategy", which is always performed when `Highs::getIis` is called.
- 1 => From dual ray, which is currently unavailable.
- 2 => From the whole LP (solving an elasticity LP repeatedly (fixing positive elastic variables at zero) until no more elastic variables are positive, and using the fixed elastic variables to determine a set of infeasible rows, for which there is a corresponding set of columns with nonzeros in those rows that form an infeasibility set (IS).
- 4 => Attempt to reduce the IS to an IIS.
- 8 => Prioritize low numbers of columns (rather than low numbers of rows) when reducing the IS.

Hence, by just setting the 2-bit, an IS is formed reliably, and at not great expense (for an LP).

Prompted by [#2653](https://github.com/ERGO-Code/HiGHS/issues/2653),
the very rare report of spurious primal infeasibilities in the optimal
solution of MIPs with large bounds on variables or constraints is
eliminated.

Prompted by [#2655](https://github.com/ERGO-Code/HiGHS/issues/2655),
[#2766](https://github.com/ERGO-Code/HiGHS/issues/2766) and
[#2744](https://github.com/ERGO-Code/HiGHS/issues/2744),
`changeRowsBounds` has been added to `highspy`.

Following PR [#2671](https://github.com/ERGO-Code/HiGHS/pull/2671),
implications gathered by the MIP solver are applied when performing a
bound change.

Prompted by [#2676](https://github.com/ERGO-Code/HiGHS/issues/2676),
the `getIis` method in `highspy` is now correct.

Following PR [#2678](https://github.com/ERGO-Code/HiGHS/pull/2678), an
option `mip_allow_cut_separation_at_nodes` (default `true`) has been
added for models where time is spent on separating cuts at branching
nodes is a considerable fraction of the solution time.

Prompted by [#2681](https://github.com/ERGO-Code/HiGHS/issues/2681),
`Highs_getPresolvedColName` and `Highs_getPresolvedRowName` have been
added to the C API.

Following PR [#2695](https://github.com/ERGO-Code/HiGHS/pull/2695),
`HPresolve::enumerateSolutions` enumerates all solutions to pure
binary constraints with up to 8 variables. This is run before probing
to avoid (expensive) probing on binary variables that could be
fixed.

Prompted by [#2696](https://github.com/ERGO-Code/HiGHS/issues/2696),
the simplex solver sets dual values of basic variables to zero,
translating this numerical error measure to nonzero residuals in the
dual equations.

Prompted by [#2705](https://github.com/ERGO-Code/HiGHS/issues/2705), a
presolved MIP can be passed from C if it contains implied integers.

Following user PR
[#2713](https://github.com/ERGO-Code/HiGHS/pull/2713), scalar
entries of the `HighsLp` class are not used after the corresponding
instance undergoes `std::move` in `highs/mip/HighsLpRelaxation.cpp`.

Prompted by [#2721](https://github.com/ERGO-Code/HiGHS/issues/2721), examples
of using solution in Python examples have been improved.

Following user PR
[#2747](https://github.com/ERGO-Code/HiGHS/pull/2747), `save_value` is
initialised wen a `HighsSimplexBadBasisChangeRecord` is created.

Following PR [#2761](https://github.com/ERGO-Code/HiGHS/pull/2761),
presolve now checks if all binary variables (from a constraint) form a
clique.

Following PR [#2762](https://github.com/ERGO-Code/HiGHS/issues/2762),
the vector is passed by reference rather than value to `freeVector` in
`highs/ipm/hipo/auxiliary/Auxiliary.h`.

Following PR [#2768](https://github.com/ERGO-Code/HiGHS/issues/2768),
the HiPO wrapper no longer makes an unnecessary copy of the LP.

Prompted by [#2769](https://github.com/ERGO-Code/HiGHS/issues/2769),
now initialising `sense` in `struct Model` of LP file reader

Compiler warnings have been fixed

All unguarded `printf` statements have been removed.


## Build changes

Added Python 3.14 wheels.

Added a CMake option `BUILD_OPENBLAS` for Windows and Linux, when `HIPO` is ON and `BUILD_OPENBLAS` is ON, OpenBLAS is downloaded and built as a subproject. The default value is OFF.

Update `rules_cuda` for the bazel build.

Filereader is now in `highs/` rather than `extern/`.

Metis, AMD and RCM are now in `extern/`. Metis is no longer an external dependency.

Binaries are now available. Standard HiGHS binaries are MIT-licensed and HiGHS with HiPO are Apache-licensed.

## Licensing

Code not written for HiGHS is now maintained in the `/extern`
directory, and `THIRD_PARTY_NOTICES.md` contains a statement of the
licenses of all such external code, some of which are permissive
non-MIT licenses.

When HiGHS is built from code, the conditions of the non-MIT licenses
are such that the resulting binaries and executable remain MIT
licensed. However, since the interior point solver HiPO makes use of
some of this external code, we believe that the MIT license is lost
when users link to our precompiled binaries. Hence there are binaries
without HiPO (MIT license), and with HiPO (Apache 2.0 license).

We believe that this is a conservative stance, and we are taking
expert advice that may allow us to relax this distinction.
