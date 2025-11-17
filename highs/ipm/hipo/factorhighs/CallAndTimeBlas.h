#ifndef FACTORHIGHS_CALL_AND_TIME_BLAS_H
#define FACTORHIGHS_CALL_AND_TIME_BLAS_H

#include "DataCollector.h"
#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

// level 1
void callAndTime_daxpy(Int64 n, double da, const double* dx, Int64 incx,
                       double* dy, Int64 incy, DataCollector& data);
void callAndTime_dcopy(Int64 n, const double* dx, Int64 incx, double* dy,
                       Int64 incy, DataCollector& data);
void callAndTime_dscal(Int64 n, const double da, double* dx, Int64 incx,
                       DataCollector& data);
void callAndTime_dswap(Int64 n, double* dx, Int64 incx, double* dy, Int64 incy,
                       DataCollector& data);

// level 2
void callAndTime_dgemv(char trans, Int64 m, Int64 n, double alpha,
                       const double* A, Int64 lda, const double* x, Int64 incx,
                       double beta, double* y, Int64 incy, DataCollector& data);
void callAndTime_dtpsv(char uplo, char trans, char diag, Int64 n,
                       const double* ap, double* x, Int64 incx,
                       DataCollector& data);
void callAndTime_dtrsv(char uplo, char trans, char diag, Int64 n,
                       const double* A, Int64 lda, double* x, Int64 incx,
                       DataCollector& data);
void callAndTime_dger(Int64 m, Int64 n, double alpha, const double* x,
                      Int64 incx, const double* y, Int64 incy, double* A,
                      Int64 lda, DataCollector& data);

// level 3
void callAndTime_dgemm(char transa, char transb, Int64 m, Int64 n, Int64 k,
                       double alpha, const double* A, Int64 lda,
                       const double* B, Int64 ldb, double beta, double* C,
                       Int64 ldc, DataCollector& data);
void callAndTime_dsyrk(char uplo, char trans, Int64 n, Int64 k, double alpha,
                       const double* a, Int64 lda, double beta, double* c,
                       Int64 ldc, DataCollector& data);
void callAndTime_dtrsm(char side, char uplo, char trans, char diag, Int64 m,
                       Int64 n, double alpha, const double* a, Int64 lda,
                       double* b, Int64 ldb, DataCollector& data);

}  // namespace hipo

#endif