#include "AutoDetect.h"

namespace hipo {

extern "C" {
int64_t cblas_isamax(const int64_t N, const float* X, const int64_t incX);
}
BlasIntegerModel getBlasIntegerModel() {
  // Test if BLAS uses 32-bit integers (LP64) or 64-bit integers (ILP64) at
  // runtime. Inspired by libblastrampoline's autodetection.c

  // Even though isamax is declared to use 64-bit integers, it may actually
  // use 32-bit integers. If a negative number is passed as first argument,
  // isamax returns 0. If the correct value of 3 is passed, it returns 2
  // instead.

  static BlasIntegerModel blas_model = BlasIntegerModel::not_set;

  if (blas_model == BlasIntegerModel::not_set) {
    // This is a very negative 64-bit number, but it is just 3 if only the lower
    // 32 bits are used.
    const int64_t n = 0xffffffff00000003;

    const float X[3] = {1.0f, 2.0f, 3.0f};

    const int64_t incx = 1;
    int64_t max_idx = cblas_isamax(n, X, incx);

    // Ignore potential upper 32 bits of the result
    max_idx = max_idx & 0xffffffff;

    if (max_idx == 0) {
      // isamax read negative n and returned 0, so it's using ilp64
      blas_model = BlasIntegerModel::ilp64;

    } else if (max_idx == 2) {
      // isamax read correct n and returned 2, so it's using lp64
      blas_model = BlasIntegerModel::lp64;

    } else {
      // something went wrong
      blas_model = BlasIntegerModel::unknown;
    }
  }

  return blas_model;
}
std::string getBlasIntegerModelString() {
  BlasIntegerModel blas_model = getBlasIntegerModel();

  switch (blas_model) {
    case BlasIntegerModel::not_set:
      return "Not set";

    case BlasIntegerModel::unknown:
      return "Unknown";

    case BlasIntegerModel::lp64:
      return "LP64";

    case BlasIntegerModel::ilp64:
      return "ILP64";
  }
}

}  // namespace hipo