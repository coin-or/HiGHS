#include "AutoDetect.h"

#include <stdint.h>

#define IDXTYPEWIDTH 64
#include "metis.h"

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

int getMetisIntegerType() {
  idx_t options[METIS_NOPTIONS];
  for (int i = 0; i < METIS_NOPTIONS; ++i) options[i] = -1;

  // if Metis is using 32-bit internally, this should set ptype to 0 and objtype
  // to 1, which should trigger an error. If it uses 64-bit then everything
  // should be fine.
  options[METIS_OPTION_PTYPE] = 1;

  idx_t n = 3;
  idx_t ptr[4] = {0, 2, 4, 6};
  idx_t rows[6] = {1, 2, 0, 2, 0, 1};
  idx_t perm[3], iperm[3];

  idx_t status = METIS_NodeND(&n, ptr, rows, NULL, options, perm, iperm);

  int metis_int = 0;
  if (status == METIS_OK) {
    if (perm[0] != 0 || perm[1] != 1 || perm[2] != 2)
      metis_int = -1;
    else
      metis_int = 64;
  } else if (status == METIS_ERROR_INPUT) {
    metis_int = 32;
  } else {
    metis_int = -1;
  }

  return metis_int;
}
}  // namespace hipo