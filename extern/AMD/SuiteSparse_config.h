//------------------------------------------------------------------------------
// SuiteSparse_config/SuiteSparse_config.h: common utilites for SuiteSparse
//------------------------------------------------------------------------------

// SuiteSparse_config, Copyright (c) 2012-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

// Configuration file for SuiteSparse: a Suite of Sparse matrix packages: AMD,
// COLAMD, CCOLAMD, CAMD, CHOLMOD, UMFPACK, CXSparse, SuiteSparseQR, ParU, ...

// The SuiteSparse_config.h file is configured by CMake to be specific to the
// C/C++ compiler and BLAS library being used for SuiteSparse.  The original
// file is SuiteSparse_config/SuiteSparse_config.h.in.  Do not edit the
// SuiteSparse_config.h file directly.

#ifndef SUITESPARSE_CONFIG_H
#define SUITESPARSE_CONFIG_H

//------------------------------------------------------------------------------
// SuiteSparse-wide ANSI C11 #include files
//------------------------------------------------------------------------------

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>

void *SuiteSparse_malloc    // pointer to allocated block of memory
(
    size_t nitems,          // number of items to malloc (>=1 is enforced)
    size_t size_of_item     // sizeof each item
) ;

void *SuiteSparse_free      // always returns NULL
(
    void *p                 // block to free
) ;

// SuiteSparse printf macro
#define SUITESPARSE_PRINTF(params)                          \
{                                                           \
                                                            \
        (void) printf params ;                              \
                                                            \
}

//==============================================================================
// SuiteSparse version
//==============================================================================

// SuiteSparse is not a package itself, but a collection of packages, some of
// which must be used together (UMFPACK requires AMD, CHOLMOD requires AMD,
// COLAMD, CAMD, and CCOLAMD, etc).  A version number is provided here for the
// collection itself, which is also the version number of SuiteSparse_config.

int SuiteSparse_version     // returns SUITESPARSE_VERSION
(
    // output, not defined on input.  Not used if NULL.  Returns
    // the three version codes in version [0..2]:
    // version [0] is SUITESPARSE_MAIN_VERSION
    // version [1] is SUITESPARSE_SUB_VERSION
    // version [2] is SUITESPARSE_SUBSUB_VERSION
    int version [3]
) ;

#define SUITESPARSE_HAS_VERSION_FUNCTION

#define SUITESPARSE_DATE "Nov 4, 2025"
#define SUITESPARSE_MAIN_VERSION    7
#define SUITESPARSE_SUB_VERSION     12
#define SUITESPARSE_SUBSUB_VERSION  1

// version format x.y
#define SUITESPARSE_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define SUITESPARSE_VERSION SUITESPARSE_VER_CODE(7, 12)

// version format x.y.z
#define SUITESPARSE__VERCODE(main,sub,patch) \
    (((main)*1000ULL + (sub))*1000ULL + (patch))
#define SUITESPARSE__VERSION SUITESPARSE__VERCODE(7,12,1)

#endif

