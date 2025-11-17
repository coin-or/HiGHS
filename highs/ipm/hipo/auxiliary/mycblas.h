#ifndef FACTORHIGHS_MY_CBLAS_H
#define FACTORHIGHS_MY_CBLAS_H

#include "IntConfig.h"

// Provide definition for cblas functions
// Based on Netlib implementation

// NB:
// Blas is declared as using 64-bit integers.
// The library that gets actually linked may use 32- or 64-bit integers,
// depending on the integer model.

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };

#ifdef __cplusplus
extern "C" {
#endif

// level 1

void cblas_daxpy(const hipo::Int64 n, const double alpha, const double* x,
                 const hipo::Int64 incx, double* y, const hipo::Int64 incy);
void cblas_dcopy(const hipo::Int64 n, const double* x, const hipo::Int64 incx,
                 double* y, const hipo::Int64 incy);
void cblas_dscal(const hipo::Int64 n, const double alpha, double* x,
                 const hipo::Int64 incx);
void cblas_dswap(const hipo::Int64 n, double* x, const hipo::Int64 incx,
                 double* y, const hipo::Int64 incy);

// level 2

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE transa, const hipo::Int64 M,
                 const hipo::Int64 n, const double alpha, const double* A,
                 const hipo::Int64 lda, const double* x, const hipo::Int64 incx,
                 const double beta, double* y, const hipo::Int64 incy);

void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
                 const hipo::Int64 n, const double* ap, double* x,
                 const hipo::Int64 incx);

void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
                 const hipo::Int64 n, const double* a, const hipo::Int64 lda,
                 double* x, const hipo::Int64 incx);

void cblas_dger(const enum CBLAS_ORDER order, const hipo::Int64 m,
                const hipo::Int64 n, const double alpha, const double* x,
                const hipo::Int64 incx, const double* y, const hipo::Int64 incy,
                double* A, const hipo::Int64 lda);

// level 3

void cblas_dgemm(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE transa,
                 const enum CBLAS_TRANSPOSE transb, const hipo::Int64 m,
                 const hipo::Int64 n, const hipo::Int64 k, const double alpha,
                 const double* A, const hipo::Int64 lda, const double* B,
                 const hipo::Int64 ldb, const double beta, double* C,
                 const hipo::Int64 ldc);

void cblas_dsyrk(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, const hipo::Int64 n,
                 const hipo::Int64 k, const double alpha, const double* a,
                 const hipo::Int64 lda, const double beta, double* C,
                 const hipo::Int64 ldc);

void cblas_dtrsm(const enum CBLAS_ORDER order, const enum CBLAS_SIDE side,
                 const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE transa,
                 const enum CBLAS_DIAG diag, const hipo::Int64 m,
                 const hipo::Int64 n, const double alpha, const double* a,
                 const hipo::Int64 lda, double* b, const hipo::Int64 ldb);

#ifdef __cplusplus
}
#endif

#endif