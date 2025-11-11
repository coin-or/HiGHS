#ifndef FACTORHIGHS_HYBRID_SOLVE_HANDLER_H
#define FACTORHIGHS_HYBRID_SOLVE_HANDLER_H

#include "SolveHandler.h"
#include "ipm/hipo/auxiliary/Auxiliary.h"

namespace hipo {

class HybridSolveHandler : public SolveHandler {
  const std::vector<std::vector<Int>>& swaps_;
  const std::vector<std::vector<double>>& pivot_2x2_;

  const std::vector<Int>& first_child_;
  const std::vector<Int>& next_child_;
  const std::vector<Int>& first_child_reverse_;
  const std::vector<Int>& next_child_reverse_;

  std::vector<std::vector<double>>& local_;

  void forwardSolve(std::vector<double>& x) const override;
  void backwardSolve(std::vector<double>& x) const override;
  void diagSolve(std::vector<double>& x) const override;

  void parForwardSolve(std::vector<double>& x) override;

  void processSupernode(Int sn, const std::vector<double>& x, bool parallelise);
  void spawnNode(Int sn, const std::vector<double>& x,
                 const TaskGroupSpecial& tg, bool do_spawn = true);
  void syncNode(Int sn, const TaskGroupSpecial& tg);

 public:
  HybridSolveHandler(const Symbolic& S,
                     const std::vector<std::vector<double>>& sn_columns,
                     const std::vector<std::vector<Int>>& swaps,
                     const std::vector<std::vector<double>>& pivot_2x2,
                     const std::vector<Int>& fc, const std::vector<Int>& nc,
                     const std::vector<Int>& fcr, const std::vector<Int>& ncr,
                     std::vector<std::vector<double>>& local);
};

}  // namespace hipo

#endif