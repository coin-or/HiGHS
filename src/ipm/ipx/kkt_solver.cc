#include "ipm/ipx/kkt_solver.h"
#include "ipm/ipx/timer.h"

namespace ipx {

void KKTSolver::Factorize(Iterate* pt, Info* info) {
    Timer timer;
    _Factorize(pt, info);
    info->time_kkt_factorize += timer.Elapsed();
}

void KKTSolver::Solve(const Vector& a, const Vector& b, double tol,
                      Vector& x, Vector& y, Info* info) {
    Timer timer;
    _Solve(a, b, tol, x, y, info);
    info->time_kkt_solve += timer.Elapsed();
}

Int KKTSolver::iterSum() const { return _iterSum(); }
Int KKTSolver::iterMax() const { return _iterMax(); }
Int KKTSolver::basis_changes() const { return _basis_changes(); }
Int KKTSolver::matrix_nz() const {return _matrix_nz(); }
Int KKTSolver::invert_nz() const {return _invert_nz(); }
const Basis* KKTSolver::basis() const { return _basis(); }

}  // namespace ipx
