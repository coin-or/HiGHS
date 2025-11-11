#include "Numeric.h"

#include "DataCollector.h"
#include "FactorHiGHSSettings.h"
#include "HybridSolveHandler.h"
#include "ReturnValues.h"
#include "Timing.h"
#include "ipm/hipo/auxiliary/Auxiliary.h"
#include "ipm/hipo/auxiliary/Log.h"
#include "ipm/hipo/auxiliary/VectorOperations.h"
#include "util/HighsCDouble.h"
#include "util/HighsRandom.h"

namespace hipo {

Int Numeric::solve(std::vector<double>& x) {
  // Return the number of solves performed

  if (!sn_columns_ || !S_) return kRetInvalidPointer;

  // initialise solve handler
  SH_.reset(new HybridSolveHandler(
      *S_, *sn_columns_, swaps_, pivot_2x2_, first_child_, next_child_,
      first_child_reverse_, next_child_reverse_, local_x_));

  SH_->setData(data_);

#if HIPO_TIMING_LEVEL >= 1
  Clock clock{};
#endif

#if HIPO_TIMING_LEVEL >= 2
  Clock clock_fine{};
#endif
  // permute rhs
  permuteVectorInverse(x, S_->iperm());
#if HIPO_TIMING_LEVEL >= 2
  if (data_) data_->sumTime(kTimeSolvePrepare, clock_fine.stop());
  clock_fine.start();
#endif

  // solve
  SH_->parForwardSolve(x);
  SH_->diagSolve(x);
  SH_->backwardSolve(x);

#if HIPO_TIMING_LEVEL >= 2
  if (data_) data_->sumTime(kTimeSolveSolve, clock_fine.stop());
#endif

#if HIPO_TIMING_LEVEL >= 2
  clock_fine.start();
#endif
  // unpermute solution
  permuteVector(x, S_->iperm());
#if HIPO_TIMING_LEVEL >= 2
  if (data_) data_->sumTime(kTimeSolvePrepare, clock_fine.stop());
#endif

#if HIPO_TIMING_LEVEL >= 1
  if (data_) data_->sumTime(kTimeSolve, clock.stop());
#endif

  return kRetOk;
}

void Numeric::getReg(std::vector<double>& reg) {
  // unpermute regularisation
  permuteVector(total_reg_, S_->iperm());

  reg = std::move(total_reg_);
}

void Numeric::setup() {
  assert(S_);

  if (ready_) return;

  // create linked lists of children in supernodal elimination tree
  childrenLinkedList(S_->snParent(), first_child_, next_child_);

  // create reverse linked lists of children
  first_child_reverse_ = first_child_;
  next_child_reverse_ = next_child_;
  reverseLinkedList(first_child_reverse_, next_child_reverse_);

  // allocate local space for parallel solve
  local_x_.resize(S_->sn());
  for (Int sn = 0; sn < S_->sn(); ++sn) {
    Int ldsn = S_->ptr(sn + 1) - S_->ptr(sn);
    local_x_[sn].resize(ldsn);
  }

  ready_ = true;
}

}  // namespace hipo