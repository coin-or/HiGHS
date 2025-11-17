#ifndef FACTORHIGHS_HYBRID_HYBRID_FORMAT_H
#define FACTORHIGHS_HYBRID_HYBRID_FORMAT_H

#include "DataCollector.h"
#include "FormatHandler.h"

namespace hipo {

class HybridHybridFormatHandler : public FormatHandler {
  std::vector<Int64> diag_start_;
  DataCollector& data_;

  void initFrontal() override;
  void initClique() override;
  void assembleFrontal(Int64 i, Int64 j, double val) override;
  void assembleFrontalMultiple(Int64 num, const std::vector<double>& child,
                               Int64 nc, Int64 child_sn, Int64 row, Int64 col, Int64 i,
                               Int64 j) override;
  Int64 denseFactorise(double reg_thresh) override;
  void assembleClique(const std::vector<double>& child, Int64 nc,
                      Int64 child_sn) override;
  void extremeEntries() override;

 public:
  HybridHybridFormatHandler(const Symbolic& S, Int64 sn, const Regul& regul,
                            DataCollector& data, std::vector<double>& frontal);
};

}  // namespace hipo

#endif