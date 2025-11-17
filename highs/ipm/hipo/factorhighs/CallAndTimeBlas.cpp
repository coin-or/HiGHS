#include "CallAndTimeBlas.h"

#include "DataCollector.h"
#include "DenseFact.h"
#include "FactorHiGHSSettings.h"
#include "Timing.h"
#include "ipm/hipo/auxiliary/Auxiliary.h"
#include "ipm/hipo/auxiliary/mycblas.h"

namespace hipo {

// macros to interface with CBlas
#define TRANS(x) (x) == 'N' ? CblasNoTrans : CblasTrans
#define UPLO(x) (x) == 'U' ? CblasUpper : CblasLower
#define DIAG(x) (x) == 'N' ? CblasNonUnit : CblasUnit
#define SIDE(x) (x) == 'L' ? CblasLeft : CblasRight

// level 1

void callAndTime_daxpy(Int64 n, double da, const double* dx, Int64 incx,
                       double* dy, Int64 incy, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_daxpy(n, da, dx, incx, dy, incy);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_axpy, clock.stop());
#endif
}

void callAndTime_dcopy(Int64 n, const double* dx, Int64 incx, double* dy,
                       Int64 incy, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dcopy(n, dx, incx, dy, incy);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_copy, clock.stop());
#endif
}

void callAndTime_dscal(Int64 n, const double da, double* dx, Int64 incx,
                       DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dscal(n, da, dx, incx);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_scal, clock.stop());
#endif
}

void callAndTime_dswap(Int64 n, double* dx, Int64 incx, double* dy, Int64 incy,
                       DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dswap(n, dx, incx, dy, incy);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_swap, clock.stop());
#endif
}

// level 2

void callAndTime_dgemv(char trans, Int64 m, Int64 n, double alpha,
                       const double* A, Int64 lda, const double* x, Int64 incx,
                       double beta, double* y, Int64 incy,
                       DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dgemv(CblasColMajor, TRANS(trans), m, n, alpha, A, lda, x, incx, beta,
              y, incy);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_gemv, clock.stop());
#endif
}

void callAndTime_dtpsv(char uplo, char trans, char diag, Int64 n,
                       const double* ap, double* x, Int64 incx,
                       DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dtpsv(CblasColMajor, UPLO(uplo), TRANS(trans), DIAG(diag), n, ap, x,
              incx);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_tpsv, clock.stop());
#endif
}

void callAndTime_dtrsv(char uplo, char trans, char diag, Int64 n,
                       const double* A, Int64 lda, double* x, Int64 incx,
                       DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dtrsv(CblasColMajor, UPLO(uplo), TRANS(trans), DIAG(diag), n, A, lda, x,
              incx);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_trsv, clock.stop());
#endif
}

void callAndTime_dger(Int64 m, Int64 n, double alpha, const double* x,
                      Int64 incx, const double* y, Int64 incy, double* A,
                      Int64 lda, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_ger, clock.stop());
#endif
}

// level 3

void callAndTime_dgemm(char transa, char transb, Int64 m, Int64 n, Int64 k,
                       double alpha, const double* A, Int64 lda,
                       const double* B, Int64 ldb, double beta, double* C,
                       Int64 ldc, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dgemm(CblasColMajor, TRANS(transa), TRANS(transb), m, n, k, alpha, A,
              lda, B, ldb, beta, C, ldc);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_gemm, clock.stop());
#endif
}

void callAndTime_dsyrk(char uplo, char trans, Int64 n, Int64 k, double alpha,
                       const double* A, Int64 lda, double beta, double* C,
                       Int64 ldc, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dsyrk(CblasColMajor, UPLO(uplo), TRANS(trans), n, k, alpha, A, lda,
              beta, C, ldc);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_syrk, clock.stop());
#endif
}

void callAndTime_dtrsm(char side, char uplo, char trans, char diag, Int64 m,
                       Int64 n, double alpha, const double* A, Int64 lda,
                       double* B, Int64 ldb, DataCollector& data) {
#if HIPO_TIMING_LEVEL >= 3
  Clock clock;
#endif
  cblas_dtrsm(CblasColMajor, SIDE(side), UPLO(uplo), TRANS(trans), DIAG(diag),
              m, n, alpha, A, lda, B, ldb);
#if HIPO_TIMING_LEVEL >= 3
  data.sumTime(kTimeBlas_trsm, clock.stop());
#endif
}

}  // namespace hipo
