#include "Numeric.h"

#include "DataCollector.h"
#include "FactorHighsSettings.h"
#include "HybridSolveHandler.h"
#include "ParallelHybridSolveHandler.h"
#include "ReturnValues.h"
#include "Timing.h"
#include "ipm/hipo/auxiliary/Auxiliary.h"
#include "ipm/hipo/auxiliary/VectorOperations.h"
#include "util/HighsCDouble.h"
#include "util/HighsRandom.h"

namespace hipo {

static TempTimer forwardTimer("forward");
static TempTimer backwardTimer("backward");
static TempTimer diagTimer("diag");

Int Numeric::prepare() {
  if (!sn_columns_ || !S_ || !data_ || !options_) return kRetInvalidPointer;

  serial_SH_.reset(new HybridSolveHandler(*S_, *sn_columns_, swaps_, any_swaps_,
                                          pivot_2x2_, gemv_workspace_, *data_,
                                          *options_));

  parallel_SH_.reset(new ParallelHybridSolveHandler(
      *S_, *sn_columns_, swaps_, any_swaps_, pivot_2x2_, *data_, *options_));

  // memory allocation should happen only the first time, then memory is
  // reused. No need to zero memory each time, as it is overwritten by
  // solveHandler.
  gemv_workspace_.resize(S_->largestFront());

  if (!serial_SH_ || !parallel_SH_) return kRetGeneric;

  if (S_->size() > 1e4)
    diag_SH_ = parallel_SH_;
  else
    diag_SH_ = serial_SH_;

  if (S_->size() > 1e4 && S_->solveTreeSpeedup() > 2)
    forward_SH_ = parallel_SH_;
  else
    forward_SH_ = serial_SH_;

  if (S_->size() > 1e4 && S_->solveTreeSpeedup() > 1.2)
    backward_SH_ = parallel_SH_;
  else
    backward_SH_ = serial_SH_;

  // compute which blocks of columns require swaps
  if (options_->pivoting) {
    any_swaps_.resize(S_->sn());
    const Int nb = options_->nb;
    for (Int sn = 0; sn < S_->sn(); ++sn) {
      const Int sn_size = S_->snStart(sn + 1) - S_->snStart(sn);
      const Int n_blocks = (sn_size - 1) / nb + 1;
      any_swaps_[sn].assign(n_blocks, 0);

      for (Int j = 0; j < n_blocks; ++j) {
        const Int jb = std::min(nb, sn_size - nb * j);
        for (Int i = 0; i < jb; ++i) {
          if (swaps_[sn][nb * j + i] != i) {
            any_swaps_[sn][j] = 1;
            break;
          }
        }
      }
    }
  }

  return kRetOk;
}

Int Numeric::solve(double* x) const {
  // Return the number of solves performed

  if (!serial_SH_ || !parallel_SH_) return kRetGeneric;

  HIPO_CLOCK_CREATE;

  // permute rhs
  HIPO_CLOCK_START(2);
  permuteVectorInverse(x, S_->iperm());
  HIPO_CLOCK_STOP(2, *data_, kTimeSolvePrepare);

  // solve
  HIPO_CLOCK_START(2);

  forwardTimer.start();
  forward_SH_->forwardSolve(x);
  forwardTimer.stop();

  diagTimer.start();
  diag_SH_->diagSolve(x);
  diagTimer.stop();

  backwardTimer.start();
  backward_SH_->backwardSolve(x);
  backwardTimer.stop();

  HIPO_CLOCK_STOP(2, *data_, kTimeSolveSolve);

  // unpermute solution
  HIPO_CLOCK_START(2);
  permuteVector(x, S_->iperm());
  HIPO_CLOCK_STOP(2, *data_, kTimeSolvePrepare);

  HIPO_CLOCK_STOP(1, *data_, kTimeSolve);
  return kRetOk;
}

Int Numeric::forwardSolve(double* x) const {
  if (!forward_SH_) return kRetGeneric;
  permuteVectorInverse(x, S_->iperm());
  forward_SH_->forwardSolve(x);
  return kRetOk;
}
Int Numeric::diagSolve(double* x) const {
  if (!diag_SH_) return kRetGeneric;
  diag_SH_->diagSolve(x);
  return kRetOk;
}
Int Numeric::backwardSolve(double* x) const {
  if (!backward_SH_) return kRetGeneric;
  backward_SH_->backwardSolve(x);
  permuteVector(x, S_->iperm());
  return kRetOk;
}

#define SOLVE_MULTIPLE(f)                                        \
  if (k == 1)                                                    \
    return f(x);                                                 \
  else {                                                         \
    highs::parallel::TaskGroup tg;                               \
    const Int n = S_->size();                                    \
    std::atomic<bool> fail{false};                               \
    for (Int i = 0; i < k; ++i) {                                \
      tg.spawn([=, &fail]() {                                    \
        Int status = f(&x[i * n]);                               \
        if (status) fail.store(true, std::memory_order_relaxed); \
      });                                                        \
    }                                                            \
    tg.taskWait();                                               \
    return fail;                                                 \
  }

Int Numeric::solve(double* x, Int k) const { SOLVE_MULTIPLE(solve); }
Int Numeric::forwardSolve(double* x, Int k) const {
  SOLVE_MULTIPLE(forwardSolve);
}
Int Numeric::diagSolve(double* x, Int k) const { SOLVE_MULTIPLE(diagSolve); }
Int Numeric::backwardSolve(double* x, Int k) const {
  SOLVE_MULTIPLE(backwardSolve);
}

void Numeric::getReg(double* reg) {
  std::memcpy(reg, total_reg_.data(), total_reg_.size() * sizeof(double));
}

void Numeric::inertia(Int& pos, Int& neg, Int& zero, double tol) const {
  if (!serial_SH_) return;
  serial_SH_->inertia(pos, neg, zero, tol);
}

}  // namespace hipo