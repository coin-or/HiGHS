#include "VectorOperations.h"

#include <cassert>
#include <cmath>

#include "HighsExternalApi.h"

namespace hipo {

void vectorAdd(std::vector<double>& v1, const std::vector<double>& v2,
               double alpha) {
  HighsExtras::blas::daxpy(v1.size(), alpha, v2.data(), 1, v1.data(), 1);
}

void vectorAdd(std::vector<double>& v1, double alpha,
               const std::vector<double>& v2, double beta) {
  HighsExtras::blas::dscal(v1.size(), alpha, v1.data(), 1);
  HighsExtras::blas::daxpy(v1.size(), beta, v2.data(), 1, v1.data(), 1);
}

void vectorAdd(std::vector<double>& v1, const double alpha) {
  for (Int i = 0; i < static_cast<Int>(v1.size()); ++i) {
    v1[i] += alpha;
  }
}

void vectorDivide(std::vector<double>& v1, const std::vector<double>& v2) {
  for (Int i = 0; i < static_cast<Int>(v1.size()); ++i) {
    v1[i] /= v2[i];
  }
}

void vectorScale(std::vector<double>& v1, double alpha) {
  HighsExtras::blas::dscal(v1.size(), alpha, v1.data(), 1);
}

double dotProd(const std::vector<double>& v1, const std::vector<double>& v2) {
  return HighsExtras::blas::ddot(v1.size(), v1.data(), 1, v2.data(), 1);
}

double norm2(const std::vector<double>& x) {
  return HighsExtras::blas::dnrm2(x.size(), x.data(), 1);
}

double infNorm(const std::vector<double>& x) {
  size_t index = HighsExtras::blas::idamax(x.size(), x.data(), 1);
  double value = x.empty() ? 0 : std::abs(x[index]);
  return value;
}

bool isNanVector(const std::vector<double>& x) {
  for (Int i = 0; i < static_cast<Int>(x.size()); ++i) {
    if (std::isnan(x[i])) return true;
  }
  return false;
}

bool isInfVector(const std::vector<double>& x) {
  for (Int i = 0; i < static_cast<Int>(x.size()); ++i) {
    if (std::isinf(x[i])) return true;
  }
  return false;
}

}  // namespace hipo
