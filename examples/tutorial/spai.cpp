/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
* 
*   Tutorial: Sparse approximate inverse preconditioner (only available with the OpenCL backend, experimental)
*
*/

// enable Boost.uBLAS support
#define VIENNACL_WITH_UBLAS

#ifndef NDEBUG
 #define NDEBUG
#endif

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <time.h>
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/spai.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/foreach.hpp"
#include "boost/tokenizer.hpp"

#include "vector-io.hpp"

template <typename MatrixType, typename VectorType, typename SolverTag, typename Preconditioner>
void run_solver(MatrixType const & A, VectorType const & b, SolverTag const & solver_tag, Preconditioner const & precond)
{
    VectorType result = viennacl::linalg::solve(A, b, solver_tag, precond);
    std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
    VectorType residual = viennacl::linalg::prod(A, result);
    residual -= b;
    std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(b) << std::endl;
}


int main (int, const char **)
{
    typedef float               ScalarType;
    typedef boost::numeric::ublas::compressed_matrix<ScalarType>        MatrixType;
    typedef boost::numeric::ublas::vector<ScalarType>                   VectorType;
    typedef viennacl::compressed_matrix<ScalarType>                     GPUMatrixType;
    typedef viennacl::vector<ScalarType>                                GPUVectorType;
  
    MatrixType M;

    //
    // Read system matrix from file
    //
    if (!viennacl::io::read_matrix_market_file(M, "../examples/testdata/mat65k.mtx"))
    {
      std::cerr<<"ERROR: Could not read matrix file " << std::endl;
      exit(EXIT_FAILURE);
    }
    
    std::cout << "Size of matrix: " << M.size1() << std::endl;
    std::cout << "Avg. Entries per row: " << M.nnz() / static_cast<double>(M.size1()) << std::endl;
    
    //
    // Use uniform load vector:
    //
    VectorType rhs(M.size2());
    for (std::size_t i=0; i<rhs.size(); ++i)
      rhs(i) = 1;

    GPUMatrixType  gpu_M(M.size1(), M.size2());
    GPUVectorType  gpu_rhs(M.size1());
    viennacl::copy(M, gpu_M);
    viennacl::copy(rhs, gpu_rhs);
    
    ///////////////////////////////// Tests to follow /////////////////////////////

    viennacl::linalg::bicgstab_tag solver_tag(1e-10, 50); //for simplicity and reasonably short execution times we use only 50 iterations here

    //
    // Reference: No preconditioner:
    //
    std::cout << "--- Reference 1: Pure BiCGStab on CPU ---" << std::endl;
    VectorType result = viennacl::linalg::solve(M, rhs, solver_tag);
    std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
    VectorType residual = viennacl::linalg::prod(M, result) - rhs;
    std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;

    std::cout << "--- Reference 2: Pure BiCGStab on GPU ---" << std::endl;
    GPUVectorType gpu_result = viennacl::linalg::solve(gpu_M, gpu_rhs, solver_tag);
    std::cout << " * Solver iterations: " << solver_tag.iters() << std::endl;
    GPUVectorType gpu_residual = viennacl::linalg::prod(gpu_M, gpu_result);
    gpu_residual -= gpu_rhs;
    std::cout << " * Rel. Residual: " << viennacl::linalg::norm_2(gpu_residual) / viennacl::linalg::norm_2(gpu_rhs) << std::endl;
    
    
    //
    // Reference: ILUT preconditioner:
    //
    std::cout << "--- Reference 2: BiCGStab with ILUT on CPU ---" << std::endl;
    std::cout << " * Preconditioner setup..." << std::endl;
    viennacl::linalg::ilut_precond<MatrixType> ilut(M, viennacl::linalg::ilut_tag());
    std::cout << " * Iterative solver run..." << std::endl;
    run_solver(M, rhs, solver_tag, ilut);
    
    
    //
    // Test 1: SPAI with CPU:
    //
    std::cout << "--- Test 1: CPU-based SPAI ---" << std::endl;  
    std::cout << " * Preconditioner setup..." << std::endl;
    viennacl::linalg::spai_precond<MatrixType> spai_cpu(M, viennacl::linalg::spai_tag(1e-3, 3, 5e-2));
    std::cout << " * Iterative solver run..." << std::endl;
    run_solver(M, rhs, solver_tag, spai_cpu);
    
    //
    // Test 2: FSPAI with CPU:
    //      
    std::cout << "--- Test 2: CPU-based FSPAI ---" << std::endl;  
    std::cout << " * Preconditioner setup..." << std::endl;
    viennacl::linalg::fspai_precond<MatrixType> fspai_cpu(M, viennacl::linalg::fspai_tag());
    std::cout << " * Iterative solver run..." << std::endl;
    run_solver(M, rhs, solver_tag, fspai_cpu);
    
    //
    // Test 3: SPAI with GPU:
    //      
    std::cout << "--- Test 3: GPU-based SPAI ---" << std::endl;  
    std::cout << " * Preconditioner setup..." << std::endl;
    viennacl::linalg::spai_precond<GPUMatrixType> spai_gpu(gpu_M, viennacl::linalg::spai_tag(1e-3, 3, 5e-2));
    std::cout << " * Iterative solver run..." << std::endl;
    run_solver(gpu_M, gpu_rhs, solver_tag, spai_gpu);
    
    return EXIT_SUCCESS;
}

