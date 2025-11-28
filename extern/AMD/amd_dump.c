//------------------------------------------------------------------------------
// AMD/Source/amd_dump: debug routines for AMD
//------------------------------------------------------------------------------

// AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
// Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* Debugging routines for AMD.  Not used if NDEBUG is not defined at compile-
 * time (the default).  See comments in amd_internal.h on how to enable
 * debugging.  Not user-callable.
 */

#include "amd_internal.h"

#ifndef NDEBUG

/* This global variable is present only when debugging */
amd_int amd_debug = -999 ;		/* default is no debug printing */

/* ========================================================================= */
/* === AMD_debug_init ====================================================== */
/* ========================================================================= */

/* Sets the debug print level, by reading the file debug.amd (if it exists) */

void amd_debug_init ( char *s )
{
    FILE *f ;
    f = fopen ("debug.amd", "r") ;
    if (f == (FILE *) NULL)
    {
	amd_debug = -999 ;
    }
    else
    {
	fscanf (f, amd_id, &amd_debug) ;
	fclose (f) ;
    }
    if (amd_debug >= 0)
    {
	printf ("%s: amd_debug_init, D= "amd_id"\n", s, amd_debug) ;
    }
}

/* ========================================================================= */
/* === AMD_dump ============================================================ */
/* ========================================================================= */

/* Dump AMD's data structure, except for the hash buckets.  This routine
 * cannot be called when the hash buckets are non-empty.
 */

void amd_dump (
    amd_int n,	    /* A is n-by-n */
    amd_int Pe [ ],	    /* pe [0..n-1]: index in iw of start of row i */
    amd_int Iw [ ],	    /* workspace of size iwlen, iwlen [0..pfree-1]
		     * holds the matrix on input */
    amd_int Len [ ],    /* len [0..n-1]: length for row i */
    amd_int iwlen,	    /* length of iw */
    amd_int pfree,	    /* iw [pfree ... iwlen-1] is empty on input */
    amd_int Nv [ ],	    /* nv [0..n-1] */
    amd_int Next [ ],   /* next [0..n-1] */
    amd_int Last [ ],   /* last [0..n-1] */
    amd_int Head [ ],   /* head [0..n-1] */
    amd_int Elen [ ],   /* size n */
    amd_int Degree [ ], /* size n */
    amd_int W [ ],	    /* size n */
    amd_int nel
)
{
    amd_int i, pe, elen, nv, len, e, p, k, j, deg, w, cnt, ilast ;

    if (amd_debug < 0) return ;
    ASSERT (pfree <= iwlen) ;
    AMD_DEBUG3 (("\nAMD dump, pfree: "amd_id"\n", pfree)) ;
    for (i = 0 ; i < n ; i++)
    {
	pe = Pe [i] ;
	elen = Elen [i] ;
	nv = Nv [i] ;
	len = Len [i] ;
	w = W [i] ;

	if (elen >= EMPTY)
	{
	    if (nv == 0)
	    {
		AMD_DEBUG3 (("\nI "amd_id": nonprincipal:    ", i)) ;
		ASSERT (elen == EMPTY) ;
		if (pe == EMPTY)
		{
		    AMD_DEBUG3 ((" dense node\n")) ;
		    ASSERT (w == 1) ;
		}
		else
		{
		    ASSERT (pe < EMPTY) ;
		    AMD_DEBUG3 ((" i "amd_id" -> parent "amd_id"\n", i, FLIP (Pe[i])));
		}
	    }
	    else
	    {
		AMD_DEBUG3 (("\nI "amd_id": active principal supervariable:\n",i));
		AMD_DEBUG3 (("   nv(i): "amd_id"  Flag: %d\n", nv, (nv < 0))) ;
		ASSERT (elen >= 0) ;
		ASSERT (nv > 0 && pe >= 0) ;
		p = pe ;
		AMD_DEBUG3 (("   e/s: ")) ;
		if (elen == 0) AMD_DEBUG3 ((" : ")) ;
		ASSERT (pe + len <= pfree) ;
		for (k = 0 ; k < len ; k++)
		{
		    j = Iw [p] ;
		    AMD_DEBUG3 (("  "amd_id"", j)) ;
		    ASSERT (j >= 0 && j < n) ;
		    if (k == elen-1) AMD_DEBUG3 ((" : ")) ;
		    p++ ;
		}
		AMD_DEBUG3 (("\n")) ;
	    }
	}
	else
	{
	    e = i ;
	    if (w == 0)
	    {
		AMD_DEBUG3 (("\nE "amd_id": absorbed element: w "amd_id"\n", e, w)) ;
		ASSERT (nv > 0 && pe < 0) ;
		AMD_DEBUG3 ((" e "amd_id" -> parent "amd_id"\n", e, FLIP (Pe [e]))) ;
	    }
	    else
	    {
		AMD_DEBUG3 (("\nE "amd_id": unabsorbed element: w "amd_id"\n", e, w)) ;
		ASSERT (nv > 0 && pe >= 0) ;
		p = pe ;
		AMD_DEBUG3 ((" : ")) ;
		ASSERT (pe + len <= pfree) ;
		for (k = 0 ; k < len ; k++)
		{
		    j = Iw [p] ;
		    AMD_DEBUG3 (("  "amd_id"", j)) ;
		    ASSERT (j >= 0 && j < n) ;
		    p++ ;
		}
		AMD_DEBUG3 (("\n")) ;
	    }
	}
    }

    /* this routine cannot be called when the hash buckets are non-empty */
    AMD_DEBUG3 (("\nDegree lists:\n")) ;
    if (nel >= 0)
    {
	cnt = 0 ;
	for (deg = 0 ; deg < n ; deg++)
	{
	    if (Head [deg] == EMPTY) continue ;
	    ilast = EMPTY ;
	    AMD_DEBUG3 ((amd_id": \n", deg)) ;
	    for (i = Head [deg] ; i != EMPTY ; i = Next [i])
	    {
		AMD_DEBUG3 (("   "amd_id" : next "amd_id" last "amd_id" deg "amd_id"\n",
		    i, Next [i], Last [i], Degree [i])) ;
		ASSERT (i >= 0 && i < n && ilast == Last [i] &&
		    deg == Degree [i]) ;
		cnt += Nv [i] ;
		ilast = i ;
	    }
	    AMD_DEBUG3 (("\n")) ;
	}
	ASSERT (cnt == n - nel) ;
    }

}

#endif
