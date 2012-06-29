/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

//
// include necessary system headers
//
#include <iostream>

//#define NDEBUG

//
// Include MTL4 headers
//
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_HAVE_MTL4 1

//
// ViennaCL includes
//
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/io/matrix_market.hpp"


// Some helper functions for this tutorial:
#include "Random.hpp"
#include "vector-io.hpp"

//
// The following function is just a hel
//
/*template <typename MTLMatrixType>
void read_system(MTLMatrixType & matrix)
{
  typedef typename MTLMatrixType::value_type    value_type;
  
  std::vector<std::map<unsigned int, value_type> >  stl_matrix(mtl::num_rows(matrix), mtl::num_cols(matrix));
  
  viennacl::tools::sparse_matrix_adapter<value_type> adapted_stl_matrix(stl_matrix);
  
  #ifdef _MSC_VER
  if (!viennacl::io::read_matrix_market_file(adapted_stl_matrix, "../../examples/testdata/mat65k.mtx"))
  #else
  if (!viennacl::io::read_matrix_market_file(adapted_stl_matrix, "../examples/testdata/mat65k.mtx"))
  #endif
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }

  // Now shift to MTL matrix:
  
  mtl::matrix::inserter< MTLMatrixType >  ins(matrix);
  typedef typename mtl::Collection<MTLMatrixType>::value_type  ValueType;
  
  typedef viennacl::tools::sparse_matrix_adapter<value_type>::iterator1  Iterator1;
  for (Iterator1 it = adapted_stl_matrix.begin1();
                 it != adapted_stl_matrix.end1();
               ++it)
  {
    for (typename Iterator1::iterator it2 = it.begin();
                                      it2 != it.end();
                                    ++it2)
    {
      ins(it2.index1(), it2.index2() << ValueType(*it);
    }
  }
} */



int main(int, char *[])
{
  typedef double    ScalarType;
  
  mtl::compressed2D<ScalarType> mtl4_matrix;
  mtl4_matrix.change_dim(65025, 65025);
  set_to_zero(mtl4_matrix);  
  
  mtl::dense_vector<ScalarType> mtl4_rhs(65025, 0.0);
  mtl::dense_vector<ScalarType> mtl4_result(65025, 0.0);
  mtl::dense_vector<ScalarType> mtl4_ref_result(65025, 0.0);
  mtl::dense_vector<ScalarType> mtl4_residual(65025, 0.0);
  
  //
  // Read system from file
  //

  #ifdef _MSC_VER
  mtl::io::matrix_market_istream("../../examples/testdata/mat65k.mtx") >> mtl4_matrix;
  #else
  mtl::io::matrix_market_istream("../examples/testdata/mat65k.mtx") >> mtl4_matrix;
  #endif
    
  #ifdef _MSC_VER
  if (!readVectorFromFile("../../examples/testdata/result65025.txt", mtl4_ref_result))
  #else
  if (!readVectorFromFile("../examples/testdata/result65025.txt", mtl4_ref_result))
  #endif
  {
    std::cout << "Error reading Result file" << std::endl;
    return 0;
  }
  
  //
  //CG solver:
  //
  std::cout << "----- Running CG -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::cg_tag());

  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;
  
  //
  //BiCGStab solver:
  //
  std::cout << "----- Running BiCGStab -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::bicgstab_tag());
  
  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

  //GMRES solver:
  std::cout << "----- Running GMRES -----" << std::endl;
  mtl4_result = viennacl::linalg::solve(mtl4_matrix, mtl4_rhs, viennacl::linalg::gmres_tag());
  
  mtl4_residual = mtl4_matrix * mtl4_result - mtl4_rhs;
  std::cout << "Relative residual: " << viennacl::linalg::norm_2(mtl4_residual) / viennacl::linalg::norm_2(mtl4_rhs) << std::endl;

}

