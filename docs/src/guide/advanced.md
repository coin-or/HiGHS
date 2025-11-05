# [Advanced features](@id guide-advanced)


## Simplex tableau data

HiGHS has a suite of methods for operations with the invertible
representation of the current basis matrix ``B``. To use
these requires knowledge of the corresponding (ordered) basic
variables. This is obtained using the
method
`getBasicVariables`
, with non-negative values being
columns and negative values corresponding to row indices plus one [so
-1 indicates row 0]. Methods
`getBasisInverseRow`
and
`getBasisInverseCol`
yield a specific row or column
of ``B^{-1}``. Methods
`getBasisSolve`
and
`getBasisTransposeSolve`
yield the solution
of ``Bx=b`` and ``B^{T}x=b`` respectively. Finally, the
methods
`getReducedRow`
and
`getReducedColumn`
yield a specific row or column of ``B^{-1}A``. In all cases,
HiGHS can return the number and indices of the nonzeros in the result.

## Irreducible infeasibility system (IIS) detection(@id highs-iis)

An Irreducible infeasibility system (IIS) consists of a set of
variables and a set of constraints in a model, together with
variable/constraint bound information, that cannot be satisfied (so is
infeasible). It is irreducible in that if any constraint or variable
bound is removed, then the system can be satisfied (so is feasible).

HiGHS has an IIS facility that is under development. Currently it can only be used for LPs.

## IIS-related methods in the `Highs` class

- `const HighsLp& getIisLp()`: Return a const reference to the internal IIS LP instance
- `HighsStatus getIis(HighsIis& iis)`: Try to find an IIS for the incumbent model. Gets the internal [`HighsIis`](@ref highs-iis-class) instance, returning `HighsStatus::kError` if the calculation failed. Note that if the incumbent model is found to be feasible, this is a "success", and `HighsStatus::kOk` is returned.
- ` HighsStatus writeIisModel(const std::string& filename = "")`: Write out the internal IIS LP instance to a file.



