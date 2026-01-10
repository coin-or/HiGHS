# `Highs::run()`

`Highs::run()` has evolved a great deal since it was first created to
"solve" the LP in the `HighsLp` instance `Highs::lp_`. When the
multiple objective code was added, `Highs::optimizeModel()` inherited
the content of `Highs::run()` so that a single call to`Highs::run()`
could perform multiple optimizations.

Other developments that have been implemented inelegantly are

### Actioning executable run-time options via `Highs::run()`

As [#2269](https://github.com/ERGO-Code/HiGHS/issues/2269)
highlighted, users of `cvxpy` can only execute `Highs::run()`, so the
following actions that were previously in `app/RunHighs.cpp`, are now
performed in `Highs::run()`

- Read from a solution and/or basis file
- Write out the model
- Write out the IIS model
- Write out the solution and/or basis file. 

There is still one action in`app/RunHighs.cpp` that should be performed in `Highs::run()`

- Write out the presolved model

These "HiGHS files" actions must only be performed at the "top level"
of `Highs::run()`, and this is acheived by caching the file options in
the `Highs` class and clearing them from options_ so that they aren't
applied at lower level calls to `Highs::run()`. They are then restored
before returning from `Highs::run()`.

### Performing user scaling

User objective and/or bound scaling is performed before assess
excessive problem data and suggesting user objective and bound
scaling. These user scaling actions must only be performed at the "top
level" of `Highs::run()`, and this is acheived by caching the user
scaling options in the `Highs` class and clearing them from options_
so that they aren't applied at lower level calls to `Highs::run()`. If
user scaling has been applied in a call to `Highs::run()`, it is
unapplied and the option values restored before returning from
`Highs::run()`.

### Applying "mods"

The `HighsLp` class contains data values and structures that cannot be handled explicitly by the solvers.

- If a variable has an excessivly large objective cost, this is
  interpreted as being infinte, and handled in
  `Highs::handleInfCost()` by fixing the variable at its lower or
  upper bound (when finite) according to the sign of the cost and the
  sense of the optimization, and zeroing the cost. After solving the
  problem, the cost and bounds must be restored.

- If a variable is of type `HighsVarType::kSemiContinuous` or
  `HighsVarType::kSemiInteger`, it is assessed in
  `assessSemiVariables` (in `HighsLpUtils.cpp`) before reformulation
  in `withoutSemiVariables` (in `HighsLpUtils.cpp`)

  - If it is not strictly "semi" it is set to`HighsVarType::kContinuous` or `HighsVarType::kInteger`
  - If its lower bound is not positive, it is deemed to be illegal
  - If its upper bound is larger than `kMaxSemiVariableUpper` then,
    depending on the lower bound, if it is possible to reformulate it the
    upper bound is set to `kMaxSemiVariableUpper` (it is said to be "tightened". 
    Otherwise, it is deemed to be illegal

These modifications are currently performed in `Highs::run()`, with
very careful code to ensure that they are removed before returning
from `Highs::run()`.

With the plan to allow indicator constraints and SOS as generalised
disjunctive forms that will be reformulated, the handling of "mods"
needs to be refactored!

## The trigger

The inelegance of `Highs::run()` (and `Highs::optimizeModel()`) was
exposed by
[\#2635](https://github.com/ERGO-Code/HiGHS/issues/2635). Both methods
need to be refactored. Firstly, `Highs::run()` must be refactored into
the following set of nested methods.

### `Highs::runFromExe()`

This "outer" layer should contain just the "HiGHS files" actions that were
previously in `app/RunHighs.cpp`, so only available to users who could
run the executable. 

### `Highs::runUserScaling()`

The next layer should handle user scaling

### `Highs::optimizeHighs()`

The next layer optimizes the problem defined in the `Highs` class with
respect to the (remaining) options. Currently this will be either a
single call to optimize what's in the `HighsModel`, or a call to
`Highs::multiobjectiveSolve()` if there are multiple objectives.

### `Highs::optimizeModel()`

The next layer just optimizes what's in the `HighsModel`, and is the current `Highs::optimizeModel()`

## Observations

- `Highs::run()` is (of course) retained, but should just return the
  value of `Highs::runFromExe()`

- Although there is no overhead within the nest of methods when
  solving an LP without file or user scaling options, calling
  `Highs::run()` from the MIP solver is opaque, so these calls
  should be to `Highs::optimizeModel()`

