#ifndef HIPO_AUXILIARY_H
#define HIPO_AUXILIARY_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

void counts2Ptr(std::vector<Int64>& ptr, std::vector<Int64>& w);
void inversePerm(const std::vector<Int64>& perm, std::vector<Int64>& iperm);
void subtreeSize(const std::vector<Int64>& parent, std::vector<Int64>& sizes);
void transpose(const std::vector<Int64>& ptr, const std::vector<Int64>& rows,
               std::vector<Int64>& ptrT, std::vector<Int64>& rowsT);
void transpose(const std::vector<Int64>& ptr, const std::vector<Int64>& rows,
               const std::vector<double>& val, std::vector<Int64>& ptrT,
               std::vector<Int64>& rowsT, std::vector<double>& valT);
void childrenLinkedList(const std::vector<Int64>& parent,
                        std::vector<Int64>& head, std::vector<Int64>& next);
void reverseLinkedList(std::vector<Int64>& head, std::vector<Int64>& next);
void dfsPostorder(Int64 node, Int64& start, std::vector<Int64>& head,
                  const std::vector<Int64>& next, std::vector<Int64>& order);
void processEdge(Int64 j, Int64 i, const std::vector<Int64>& first,
                 std::vector<Int64>& maxfirst, std::vector<Int64>& delta,
                 std::vector<Int64>& prevleaf, std::vector<Int64>& ancestor);
Int64 getDiagStart(Int64 n, Int64 k, Int64 nb, Int64 n_blocks,
                   std::vector<Int64>& start, bool triang = false);

template <typename T>
void permuteVector(std::vector<T>& v, const std::vector<Int64>& perm) {
  // Permute vector v according to permutation perm.
  std::vector<T> temp_v(v);
  for (Int64 i = 0; i < v.size(); ++i) v[i] = temp_v[perm[i]];
}

template <typename T>
void permuteVectorInverse(std::vector<T>& v, const std::vector<Int64>& iperm) {
  // Permute vector v according to inverse permutation iperm.
  std::vector<T> temp_v(v);
  for (Int64 i = 0; i < v.size(); ++i) v[iperm[i]] = temp_v[i];
}

template <typename T>
void printTest(const std::vector<T>& v, const std::string s) {
  std::ofstream out_file;
  char name[80];
  snprintf(name, 80, "%s.txt", s.c_str());
  out_file.open(name);
  for (T i : v) {
    out_file << std::setprecision(16) << i << '\n';
  }
  out_file.close();
}

class Clock {
  std::chrono::high_resolution_clock::time_point t0;

 public:
  Clock();
  void start();
  double stop() const;
};

}  // namespace hipo

#endif
