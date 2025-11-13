#ifndef FACTORHIGHS_HYBRID_PACKED_FORMAT_H
#define FACTORHIGHS_HYBRID_PACKED_FORMAT_H

#include "DataCollector.h"
#include "FormatHandler.h"

namespace hipo {

class HybridPackedFormatHandler : public FormatHandler {
  std::vector<int> diag_start_;
  DataCollector& data_;

  void initFrontal() override;
  void initClique() override;
  void assembleFrontal(int i, int j, double val) override;
  void assembleFrontalMultiple(int num, const std::vector<double>& child,
                               int nc, int child_sn, int row, int col, int i,
                               int j) override;
  int denseFactorise(double reg_thresh) override;
  void assembleClique(const std::vector<double>& child, int nc,
                      int child_sn) override;
  void extremeEntries() override;

 public:
  HybridPackedFormatHandler(const Symbolic& S, int sn, const Regul& regul,
                            DataCollector& data, std::vector<double>& frontal);
};

}  // namespace hipo

#endif