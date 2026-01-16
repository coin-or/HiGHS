#include "HybridPackedFormatHandler.h"

#include <cassert>

#include "CallAndTimeBlas.h"
#include "DataCollector.h"
#include "DenseFact.h"
#include "ipm/hipo/auxiliary/Auxiliary.h"

namespace hipo {

HybridPackedFormatHandler::HybridPackedFormatHandler(
    const Symbolic& S, Int sn, const Regul& regul, DataCollector& data,
    std::vector<double>& frontal, double* clique_ptr)
    : FormatHandler(S, sn, regul, frontal, clique_ptr), data_{data} {
  // initialize frontal and clique
  initFrontal();

  // if CliqueStack is used, clique_ptr already points to a valid region of
  // memory for the clique. Otherwise, allocate it locally.
  if (!clique_ptr_) initClique();
}

void HybridPackedFormatHandler::initFrontal() {
  const Int n_blocks = (sn_size_ - 1) / nb_ + 1;
  diag_start_.resize(n_blocks);
  Int64 frontal_size = getDiagStart(ldf_, sn_size_, nb_, n_blocks, diag_start_);
  frontal_.assign(frontal_size + extra_space_frontal, 0.0);
  // NB: extra_space_frontal is not strictly needed. However, it removes some
  // weird problem on windows in debug. Who knows what's happening...

  // frontal_ is actually allocated just the first time, then the memory is
  // reused from the previous factorisations and just initialised.
}

void HybridPackedFormatHandler::initClique() {
  clique_.resize(S_->cliqueSize(sn_));

  // If the clique size is zero, do not access the underlying pointer. This
  // causes strange issues on windows. It's not a problem if clique_ptr_ remains
  // null, because it will never be used in that case.
  if (!clique_.empty()) clique_ptr_ = clique_.data();
}

void HybridPackedFormatHandler::assembleFrontal(Int i, Int j, double val) {
  Int block = j / nb_;
  Int ldb = ldf_ - block * nb_;
  Int ii = i - block * nb_;
  Int jj = j - block * nb_;
  frontal_[diag_start_[block] + ii + ldb * jj] = val;
}

Int HybridPackedFormatHandler::denseFactorise(double reg_thresh) {
  Int status;

  // either clique is valid, or clique is not needed
  assert(clique_ptr_ || ldf_ == sn_size_);

  status = denseFactFP2FH(frontal_.data(), ldf_, sn_size_, nb_, data_);
  if (status) return status;

  // find the position within pivot_sign corresponding to this supernode
  Int sn_start = S_->snStart(sn_);
  const Int* pivot_sign = &S_->pivotSign().data()[sn_start];

  status = denseFactFH('P', ldf_, sn_size_, nb_, frontal_.data(), clique_ptr_,
                       pivot_sign, reg_thresh, regul_, local_reg_.data(),
                       swaps_.data(), pivot_2x2_.data(), S_->parNode(), data_);

  return status;
}

void HybridPackedFormatHandler::assembleChild(Int child_sn,
                                              const double* child) {
  const Int child_begin = S_->snStart(child_sn);
  const Int child_end = S_->snStart(child_sn + 1);
  const Int child_sn_size = child_end - child_begin;
  const Int child_clique_size =
      S_->ptr(child_sn + 1) - S_->ptr(child_sn) - child_sn_size;

  // go through the columns of the contribution of the child
  for (Int col = 0; col < child_clique_size; ++col) {
    // relative index of column in the frontal matrix
    const Int j = S_->relindClique(child_sn, col);

    Int row = col;
    while (row < child_clique_size) {
      // relative index of the entry in the matrix frontal
      const Int i = S_->relindClique(child_sn, row);

      // how many entries to sum
      const Int consecutive = S_->consecutiveSums(child_sn, row);

      // information to access child
      const Int block_child = col / nb_;
      const Int row_child = row - block_child * nb_;
      const Int col_child = col - block_child * nb_;
      const Int ld_child = child_clique_size - nb_ * block_child;
      const Int64 start_block_child =
          S_->cliqueBlockStart(child_sn, block_child);

      if (j < sn_size_) {
        // assemble entries, from (row,col) in child, to (i,j) in frontal

        const Int block_frontal = j / nb_;
        const Int row_frontal = i - block_frontal * nb_;
        const Int col_frontal = j - block_frontal * nb_;
        const Int ld_frontal = ldf_ - block_frontal * nb_;

        callAndTime_daxpy(
            consecutive, 1.0,
            &child[start_block_child + row_child + ld_child * col_child], 1,
            &frontal_[diag_start_[block_frontal] + row_frontal +
                      ld_frontal * col_frontal],
            1, data_);

      } else {
        // assemble entries, from (row,col) in child, to (rel_i,rel_j) in clique

        const Int rel_j = j - sn_size_;
        const Int rel_i = i - sn_size_;
        const Int block_clique = rel_j / nb_;
        const Int row_clique = rel_i - block_clique * nb_;
        const Int col_clique = rel_j - block_clique * nb_;
        const Int ld_clique = ldc_ - nb_ * block_clique;
        const Int64 start_block_clique =
            S_->cliqueBlockStart(sn_, block_clique);

        callAndTime_daxpy(
            consecutive, 1.0,
            &child[start_block_child + row_child + ld_child * col_child], 1,
            &clique_ptr_[start_block_clique + row_clique +
                         ld_clique * col_clique],
            1, data_);
      }

      row += consecutive;
    }
  }
}

void HybridPackedFormatHandler::extremeEntries() {
#ifdef HIPO_COLLECT_EXPENSIVE_DATA
  double minD = std::numeric_limits<double>::max();
  double maxD = 0.0;
  double minoffD = std::numeric_limits<double>::max();
  double maxoffD = 0.0;

  // number of blocks of columns
  const Int n_blocks = (sn_size_ - 1) / nb_ + 1;

  // index to access frontal
  Int index{};

  // go through blocks of columns for this supernode
  for (Int j = 0; j < n_blocks; ++j) {
    // number of columns in the block
    const Int jb = std::min(nb_, sn_size_ - nb_ * j);

    for (Int k = 0; k < jb; ++k) {
      // off diagonal entries
      for (Int i = 0; i < k; ++i) {
        if (frontal_[index] != 0.0) {
          minoffD = std::min(minoffD, std::abs(frontal_[index]));
          maxoffD = std::max(maxoffD, std::abs(frontal_[index]));
        }
        index++;
      }

      // diagonal entry
      minD = std::min(minD, std::abs(1.0 / frontal_[index]));
      maxD = std::max(maxD, std::abs(1.0 / frontal_[index]));

      index += jb - k;
    }

    const Int entries_left = (ldf_ - nb_ * j - jb) * jb;

    for (Int i = 0; i < entries_left; ++i) {
      if (frontal_[index] != 0.0) {
        minoffD = std::min(minoffD, std::abs(frontal_[index]));
        maxoffD = std::max(maxoffD, std::abs(frontal_[index]));
      }
      index++;
    }
  }

  data_.setExtremeEntries(minD, maxD, minoffD, maxoffD);
#endif
}

}  // namespace hipo