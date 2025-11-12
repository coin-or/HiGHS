# HiGHS irreducible infeasibility system (IIS) facility

Further to documentation in https://ergo-code.github.io/HiGHS/stable/guide/advanced/

The IIS search is rooted in `Highs::getIisInterface()`, which first
checks whether the `Highs::model_status_` is
`HighsModelStatus::kOptimal` or `HighsModelStatus::kUnbounded`, in
which case the model is feasible so no IIS exists. Otherwise, the
trivial check (for inconsistent bounds or empty infeasible rows) -
performed by `HighsIis::trivial()` - and infeasible rows based on row
value bounds - performed by `HighsIis::rowValueBounds()` - are
performed. If `Highs::options_.iis_strategy` is `kIisStrategyLight`
then `Highs::getIisInterface()` returns.

The "full" IIS calculation operates in two phases: after a set of
mutually infeasible rows has been identified, this is reduced to an
IIS. The set of mutually infeasible rows can be found in two ways.

Firstly, if it is known that the model is infeasible, then the simplex
solver may have identified a dual ray. If there is a dual ray then its
nonzeros correspond to a set of mutually infeasible constraints. If
there is no dual ray - as might happen if the model's infeasibility
has been identified in presolve - then the incumbent model is solved
with `Highs::options_.presolve` = "off". Unfortunately the "ray route"
is not robust, so currently switched off.

Secondly - and the only route at the moment - an elasticity filter
calculation is done to identify an infeasible subset of rows. This is
calculation performed in `HighsIis::elasticityFilter`. This method is
more general than is necessary for finding the set of mutually
infeasible rows for an IIS calculation, and can be called directly
using `Highs::feasibilityRelaxation`.

The essence of the `HighsIis::elasticityFilter` is that it allows
lower bounds, upper bounds and RHS values to be violated. There are
penalties for doing so that can be global for each of these three
cases, or local to each column lower bound, upper bound and row
bound. The "elasticity LP" is constructed by adding elastic variables
to transform the constraints from
$$
L <= Ax <= U;\qquad l <= x <= u
$$
to
$$
L <= Ax + e_L - e_U <= U;\qquad l <=  x + e_l - e_u <= u,
$$
where the elastic variables are not used if the corresponding bound is
infinite or the local/global penalty is negative. The original bounds
on the variables $x$ are removed, and the objective is the linear
function of the elastic variables given by the local/global
penalties. Note that the model modifications required to achieve this
formulation, are made by calls to methods in the `Highs` class so that
the value of any initial basis is maintained.

For the purposes of IIS calculation, each elastic variable has a 

If the original constraints cannot be satisfied, then some 

After the elasticity LP has been solved, each elasticity variable whose optimal value is positive is fixed at zero is 

