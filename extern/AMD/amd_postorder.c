//------------------------------------------------------------------------------
// AMD/Source/amd_postorder: post-order the assembly tree from AMD
//------------------------------------------------------------------------------

// AMD, Copyright (c) 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
// Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* Perform a postordering (via depth-first search) of an assembly tree. */

#include "amd_internal.h"

void amd_postorder
(
    /* inputs, not modified on output: */
    amd_int nn,		/* nodes are in the range 0..nn-1 */
    amd_int Parent [ ],	/* Parent [j] is the parent of j, or EMPTY if root */
    amd_int Nv [ ],		/* Nv [j] > 0 number of pivots represented by node j,
			 * or zero if j is not a node. */
    amd_int Fsize [ ],	/* Fsize [j]: size of node j */

    /* output, not defined on input: */
    amd_int Order [ ],	/* output post-order */

    /* workspaces of size nn: */
    amd_int Child [ ],
    amd_int Sibling [ ],
    amd_int Stack [ ]
)
{
    amd_int i, j, k, parent, frsize, f, fprev, maxfrsize, bigfprev, bigf, fnext ;

    for (j = 0 ; j < nn ; j++)
    {
	Child [j] = EMPTY ;
	Sibling [j] = EMPTY ;
    }

    /* --------------------------------------------------------------------- */
    /* place the children in link lists - bigger elements tend to be last */
    /* --------------------------------------------------------------------- */

    for (j = nn-1 ; j >= 0 ; j--)
    {
	if (Nv [j] > 0)
	{
	    /* this is an element */
	    parent = Parent [j] ;
	    if (parent != EMPTY)
	    {
		/* place the element in link list of the children its parent */
		/* bigger elements will tend to be at the end of the list */
		Sibling [j] = Child [parent] ;
		Child [parent] = j ;
	    }
	}
    }

#ifndef NDEBUG
    {
	Int nels, ff, nchild ;
	AMD_DEBUG1 (("\n\n================================ AMD_postorder:\n"));
	nels = 0 ;
	for (j = 0 ; j < nn ; j++)
	{
	    if (Nv [j] > 0)
	    {
		AMD_DEBUG1 (( ""amd_id" :  nels "amd_id" npiv "amd_id" size "amd_id
		    " parent "amd_id" maxfr "amd_id"\n", j, nels,
		    Nv [j], Fsize [j], Parent [j], Fsize [j])) ;
		/* this is an element */
		/* dump the link list of children */
		nchild = 0 ;
		AMD_DEBUG1 (("    Children: ")) ;
		for (ff = Child [j] ; ff != EMPTY ; ff = Sibling [ff])
		{
		    AMD_DEBUG1 ((amd_id" ", ff)) ;
		    ASSERT (Parent [ff] == j) ;
		    nchild++ ;
		    ASSERT (nchild < nn) ;
		}
		AMD_DEBUG1 (("\n")) ;
		parent = Parent [j] ;
		if (parent != EMPTY)
		{
		    ASSERT (Nv [parent] > 0) ;
		}
		nels++ ;
	    }
	}
    }
    AMD_DEBUG1 (("\n\nGo through the children of each node, and put\n"
		 "the biggest child last in each list:\n")) ;
#endif

    /* --------------------------------------------------------------------- */
    /* place the largest child last in the list of children for each node */
    /* --------------------------------------------------------------------- */

    for (i = 0 ; i < nn ; i++)
    {
	if (Nv [i] > 0 && Child [i] != EMPTY)
	{

#ifndef NDEBUG
	    Int nchild ;
	    AMD_DEBUG1 (("Before partial sort, element "amd_id"\n", i)) ;
	    nchild = 0 ;
	    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
	    {
		ASSERT (f >= 0 && f < nn) ;
		AMD_DEBUG1 (("      f: "amd_id"  size: "amd_id"\n", f, Fsize [f])) ;
		nchild++ ;
		ASSERT (nchild <= nn) ;
	    }
#endif

	    /* find the biggest element in the child list */
	    fprev = EMPTY ;
	    maxfrsize = EMPTY ;
	    bigfprev = EMPTY ;
	    bigf = EMPTY ;
	    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
	    {
		ASSERT (f >= 0 && f < nn) ;
		frsize = Fsize [f] ;
		if (frsize >= maxfrsize)
		{
		    /* this is the biggest seen so far */
		    maxfrsize = frsize ;
		    bigfprev = fprev ;
		    bigf = f ;
		}
		fprev = f ;
	    }
	    ASSERT (bigf != EMPTY) ;

	    fnext = Sibling [bigf] ;

	    AMD_DEBUG1 (("bigf "amd_id" maxfrsize "amd_id" bigfprev "amd_id" fnext "amd_id
		" fprev " amd_id"\n", bigf, maxfrsize, bigfprev, fnext, fprev)) ;

	    if (fnext != EMPTY)
	    {
		/* if fnext is EMPTY then bigf is already at the end of list */

		if (bigfprev == EMPTY)
		{
		    /* delete bigf from the element of the list */
		    Child [i] = fnext ;
		}
		else
		{
		    /* delete bigf from the middle of the list */
		    Sibling [bigfprev] = fnext ;
		}

		/* put bigf at the end of the list */
		Sibling [bigf] = EMPTY ;
		ASSERT (Child [i] != EMPTY) ;
		ASSERT (fprev != bigf) ;
		ASSERT (fprev != EMPTY) ;
		Sibling [fprev] = bigf ;
	    }

#ifndef NDEBUG
	    AMD_DEBUG1 (("After partial sort, element "amd_id"\n", i)) ;
	    for (f = Child [i] ; f != EMPTY ; f = Sibling [f])
	    {
		ASSERT (f >= 0 && f < nn) ;
		AMD_DEBUG1 (("        "amd_id"  "amd_id"\n", f, Fsize [f])) ;
		ASSERT (Nv [f] > 0) ;
		nchild-- ;
	    }
	    ASSERT (nchild == 0) ;
#endif

	}
    }

    /* --------------------------------------------------------------------- */
    /* postorder the assembly tree */
    /* --------------------------------------------------------------------- */

    for (i = 0 ; i < nn ; i++)
    {
	Order [i] = EMPTY ;
    }

    k = 0 ;

    for (i = 0 ; i < nn ; i++)
    {
	if (Parent [i] == EMPTY && Nv [i] > 0)
	{
	    AMD_DEBUG1 (("Root of assembly tree "amd_id"\n", i)) ;
	    k = amd_post_tree (i, k, Child, Sibling, Order, Stack
#ifndef NDEBUG
		, nn
#endif
		) ;
	}
    }
}
