#include "Symbolic.h"

#include <iostream>

#include "FactorHiGHSSettings.h"
#include "ipm/hipo/auxiliary/Log.h"

namespace hipo {

Symbolic::Symbolic() {}

void Symbolic::setParallel(bool par_tree, bool par_node) {
  parallel_tree_ = par_tree;
  parallel_node_ = par_node;
}

void Symbolic::setMetisNo2hop(bool metis_no2hop) {
  metis_no2hop_ = metis_no2hop;
}

int64_t Symbolic::nz() const { return nz_; }
double Symbolic::flops() const { return flops_; }
double Symbolic::spops() const { return spops_; }
double Symbolic::critops() const { return critops_; }
Int64 Symbolic::blockSize() const { return block_size_; }
Int64 Symbolic::size() const { return n_; }
Int64 Symbolic::sn() const { return sn_; }
double Symbolic::fillin() const { return fillin_; }
Int64 Symbolic::rows(Int64 i) const { return rows_[i]; }
Int64 Symbolic::ptr(Int64 i) const { return ptr_[i]; }
Int64 Symbolic::snStart(Int64 i) const { return sn_start_[i]; }
Int64 Symbolic::snParent(Int64 i) const { return sn_parent_[i]; }
Int64 Symbolic::relindCols(Int64 i) const { return relind_cols_[i]; }
Int64 Symbolic::relindClique(Int64 i, Int64 j) const {
  return relind_clique_[i][j];
}
Int64 Symbolic::consecutiveSums(Int64 i, Int64 j) const {
  return consecutive_sums_[i][j];
}
Int64 Symbolic::cliqueBlockStart(Int64 sn, Int64 bl) const {
  return clique_block_start_[sn][bl];
}
Int64 Symbolic::cliqueSize(Int64 sn) const {
  return clique_block_start_[sn].back();
}
bool Symbolic::parTree() const { return parallel_tree_; }
bool Symbolic::parNode() const { return parallel_node_; }
bool Symbolic::metisNo2hop() const { return metis_no2hop_; }

const std::vector<Int64>& Symbolic::ptr() const { return ptr_; }
const std::vector<Int>& Symbolic::iperm() const { return iperm_; }
const std::vector<Int64>& Symbolic::snParent() const { return sn_parent_; }
const std::vector<Int64>& Symbolic::snStart() const { return sn_start_; }
const std::vector<Int>& Symbolic::pivotSign() const { return pivot_sign_; }

static std::string memoryString(double mem) {
  std::stringstream ss;

  if (mem < 1024)
    ss << sci(mem, 0, 1) << " B";
  else if (mem < 1024 * 1024)
    ss << sci(mem / 1024, 0, 1) << " KB";
  else if (mem < 1024 * 1024 * 1024)
    ss << sci(mem / 1024 / 1024, 0, 1) << " MB";
  else
    ss << sci(mem / 1024 / 1024 / 1024, 0, 1) << " GB";

  return ss.str();
}

void Symbolic::print(const Log& log, bool verbose) const {
  std::stringstream log_stream;
  log_stream << "\nFactorisation statistics\n";
  log_stream << textline("Size:") << sci(n_, 0, 2) << '\n';
  log_stream << textline("Nnz:") << sci(nz_, 0, 2) << '\n';
  log_stream << textline("Fill-in:") << fix(fillin_, 0, 2) << '\n';
  log_stream << textline("Serial memory:") << memoryString(serial_storage_)
             << '\n';
  log_stream << textline("Flops:") << sci(flops_, 0, 1) << '\n';
  if (verbose) {
    log_stream << textline("Sparse ops:") << sci(spops_, 0, 1) << '\n';
    log_stream << textline("Critical ops:") << sci(critops_, 0, 1) << '\n';
    log_stream << textline("Max tree speedup:") << fix(flops_ / critops_, 0, 2)
               << '\n';
    log_stream << textline("Artificial nz:") << sci(artificial_nz_, 0, 1)
               << '\n';
    log_stream << textline("Artificial ops:") << sci(artificial_ops_, 0, 1)
               << '\n';
    log_stream << textline("Largest front:") << integer(largest_front_, 0)
               << '\n';
    log_stream << textline("Largest supernode:") << integer(largest_sn_, 0)
               << '\n';
    log_stream << textline("Supernodes:") << integer(sn_, 0) << '\n';
    log_stream << textline("Sn size <= 1:") << integer(sn_size_1_, 0) << '\n';
    log_stream << textline("Sn size <= 10:") << integer(sn_size_10_, 0) << '\n';
    log_stream << textline("Sn size <= 100:") << integer(sn_size_100_, 0)
               << '\n';
    log_stream << textline("Sn avg size:") << sci((double)n_ / sn_, 0, 1)
               << '\n';
  }
  log.print(log_stream);

  // Warn about large fill-in
  if (fillin_ > 50 && !metis_no2hop_) {
    log.printw(
        "Large fill-in in factorisation. Consider setting the "
        "hipo_metis_no2hop option to true\n");
  }

  log.print("\n");
}

}  // namespace hipo