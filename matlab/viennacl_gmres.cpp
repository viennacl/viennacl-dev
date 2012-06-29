/* =======================================================================
   Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
   http://www.iue.tuwien.ac.at
                             -----------------
							Matlab interface for
                     ViennaCL - The Vienna Computing Library
                             -----------------
                            
   authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
               Florian Rudolf                     flo.rudy+viennacl@gmail.com
               Josef Weinbub                      weinbub@iue.tuwien.ac.at

   license:    MIT (X11), see file LICENSE in the ViennaCL base directory

   file changelog: - June 10, 2010   New from scratch for first release
======================================================================= */

#include <math.h>
#include "mex.h"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/gmres.hpp"

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

void viennacl_gmres(double	* result,   //output vector
  		            mwIndex	* cols,    //input vector holding column jumpers
    	            mwIndex	* rows,    //input vector holding row indices
                    double *entries,
                    double *rhs,
                    mwSize     num_cols,
                    mwSize     nnzmax
		   )
{
    viennacl::vector<double>                    vcl_rhs(num_cols);
    viennacl::vector<double>                    vcl_result(num_cols);
    viennacl::compressed_matrix<double>     vcl_matrix(num_cols, num_cols);

    //convert from column-wise storage to row-wise storage
    std::vector< std::map< unsigned int, double > >  stl_matrix(num_cols);

    for (mwIndex j=0; j<num_cols; ++j)
    {
       for (mwIndex i = cols[j]; i<cols[j+1]; ++i)
         stl_matrix[rows[i]][j] = entries[i];
    }

    //now copy matrix to GPU:
    copy(stl_matrix, vcl_matrix);
    copy(rhs, rhs + num_cols, vcl_rhs.begin());
    stl_matrix.clear(); //clean up this temporary storage

	//solve it:
    vcl_result = solve(vcl_matrix,
                       vcl_rhs,
                       viennacl::linalg::gmres_tag(1e-8, 30, 20));    //relative tolerance of 1e-8, krylov space of dimension 30, 20 restarts max.

    ///////////// copy back to CPU: ///////////////////
    copy(vcl_result.begin(), vcl_result.end(), result);

    return;
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray *prhs[] )
     
{ 
    double *result; 
    double *rhs; 
    mwSize m,n, nnzmax;
    mwIndex * cols;
    mwIndex * rows;
    double * entries;
    
    /* Check for proper number of arguments */
    
    if (nrhs != 2) { 
	    mexErrMsgTxt("Two input arguments required."); 
    } else if (nlhs > 1) {
	    mexErrMsgTxt("Wrong number of output arguments."); 
    } 
    
    /* Check the dimensions of Y.  Y can be 4 X 1 or 1 X 4. */ 
    
    m = mxGetM(prhs[0]); 
    n = mxGetN(prhs[0]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||  !mxIsSparse(prhs[0]) ) { 
	    mexErrMsgTxt("viennacl_gmres requires a double precision real sparse matrix."); 
        return;
    } 
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) { 
	    mexErrMsgTxt("viennacl_gmres requires a double precision real right hand side vector.");
        return; 
    } 
    
    /* Create a vector for the return argument (the solution :-) ) */ 
    plhs[0] = mxCreateDoubleMatrix(MAX(m,n), 1, mxREAL); //return vector with 5 entries
    
    /* Assign pointers to the various parameters */ 
    result = mxGetPr(plhs[0]);
    
    cols    = mxGetJc(prhs[0]);
    rows    = mxGetIr(prhs[0]);
    entries = mxGetPr(prhs[0]);
    nnzmax  = mxGetNzmax(prhs[0]);
    rhs     = mxGetPr(prhs[1]);
        
    /* Do the actual computations in a subroutine */
    viennacl_gmres(result, cols, rows, entries, rhs, n, nnzmax);

    return;
    
}


