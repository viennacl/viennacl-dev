#ifndef VIENNACL_LINALG_HOST_BASED_DIRECT_SOLVE_HPP
#define VIENNACL_LINALG_HOST_BASED_DIRECT_SOLVE_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file viennacl/linalg/host_based/direct_solve.hpp
    @brief Implementations of dense direct triangular solvers are found here.
*/

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace host_based
    {
      
      namespace detail
      {
        //
        // Upper solve:
        //
        template <typename MatrixType1, typename MatrixType2>
        void upper_inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, bool unit_diagonal)
        {
          typedef typename MatrixType2::value_type   value_type;
          
          for (std::size_t i = 0; i < A_size; ++i)
          {
            std::size_t current_row = A_size - i - 1;
            
            for (std::size_t j = current_row + 1; j < A_size; ++j)
            {
              value_type A_element = A(current_row, j);
              for (std::size_t k=0; k < B_size; ++k)
                B(current_row, k) -= A_element * B(j, k);
            }
            
            if (!unit_diagonal)
            {
              value_type A_diag = A(current_row, current_row);
              for (std::size_t k=0; k < B_size; ++k)
                B(current_row, k) /= A_diag;
            }
          }
        }
          
        template <typename MatrixType1, typename MatrixType2>
        void inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, viennacl::linalg::unit_upper_tag)
        {
          upper_inplace_solve_matrix(A, B, A_size, B_size, true);
        }
          
        template <typename MatrixType1, typename MatrixType2>
        void inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, viennacl::linalg::upper_tag)
        {
          upper_inplace_solve_matrix(A, B, A_size, B_size, false);
        }
          
        //
        // Lower solve:
        //
        template <typename MatrixType1, typename MatrixType2>
        void lower_inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, bool unit_diagonal)
        {
          typedef typename MatrixType2::value_type   value_type;
          
          for (std::size_t i = 0; i < A_size; ++i)
          {
            for (std::size_t j = 0; j < i; ++j)
            {
              value_type A_element = A(i, j);
              for (std::size_t k=0; k < B_size; ++k)
                B(i, k) -= A_element * B(j, k);
            }
            
            if (!unit_diagonal)
            {
              value_type A_diag = A(i, i);
              for (std::size_t k=0; k < B_size; ++k)
                B(i, k) /= A_diag;
            }
          }
        }
          
        template <typename MatrixType1, typename MatrixType2>
        void inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, viennacl::linalg::unit_lower_tag)
        {
          lower_inplace_solve_matrix(A, B, A_size, B_size, true);
        }
          
        template <typename MatrixType1, typename MatrixType2>
        void inplace_solve_matrix(MatrixType1 & A, MatrixType2 & B, std::size_t A_size, std::size_t B_size, viennacl::linalg::lower_tag)
        {
          lower_inplace_solve_matrix(A, B, A_size, B_size, false);
        }
          
      }
      
      //
      // Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
      //
      
      ////////////////// upper triangular solver (upper_tag) //////////////////////////////////////
      /** @brief Direct inplace solver for triangular systems with multiple right hand sides, i.e. A \ B   (MATLAB notation)
      *
      * @param A      The system matrix
      * @param B      The matrix of row vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const M1 & A, M2 & B, SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type       * data_B = detail::extract_raw_pointer<value_type>(B);
        
        std::size_t A_start1 = viennacl::traits::start1(A);
        std::size_t A_start2 = viennacl::traits::start2(A);
        std::size_t A_inc1   = viennacl::traits::stride1(A);
        std::size_t A_inc2   = viennacl::traits::stride2(A);
        std::size_t A_size2  = viennacl::traits::size2(A);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A);
        
        std::size_t B_start1 = viennacl::traits::start1(B);
        std::size_t B_start2 = viennacl::traits::start2(B);
        std::size_t B_inc1   = viennacl::traits::stride1(B);
        std::size_t B_inc2   = viennacl::traits::stride2(B);
        std::size_t B_size2  = viennacl::traits::size2(B);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B);
        
        
        detail::matrix_array_wrapper<value_type const, typename M1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename M2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        
        detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SOLVERTAG());
      }
      
      /** @brief Direct inplace solver for triangular systems with multiple transposed right hand sides, i.e. A \ B^T   (MATLAB notation)
      *
      * @param A       The system matrix
      * @param proxy_B The proxy for the transposed matrix of row vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const M1 & A,
                    matrix_expression< const M2, const M2, op_trans> proxy_B,
                    SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
        value_type       * data_B = const_cast<value_type *>(detail::extract_raw_pointer<value_type>(proxy_B.lhs()));
        
        std::size_t A_start1 = viennacl::traits::start1(A);
        std::size_t A_start2 = viennacl::traits::start2(A);
        std::size_t A_inc1   = viennacl::traits::stride1(A);
        std::size_t A_inc2   = viennacl::traits::stride2(A);
        std::size_t A_size2  = viennacl::traits::size2(A);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(A);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(A);
        
        std::size_t B_start1 = viennacl::traits::start1(proxy_B.lhs());
        std::size_t B_start2 = viennacl::traits::start2(proxy_B.lhs());
        std::size_t B_inc1   = viennacl::traits::stride1(proxy_B.lhs());
        std::size_t B_inc2   = viennacl::traits::stride2(proxy_B.lhs());
        std::size_t B_size1  = viennacl::traits::size1(proxy_B.lhs());
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(proxy_B.lhs());
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(proxy_B.lhs());
        
        
        detail::matrix_array_wrapper<value_type const, typename M1::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename M2::orientation_category, true>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        
        detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size1, SOLVERTAG());
      }
      
      //upper triangular solver for transposed lower triangular matrices
      /** @brief Direct inplace solver for transposed triangular systems with multiple right hand sides, i.e. A^T \ B   (MATLAB notation)
      *
      * @param proxy_A  The transposed system matrix proxy
      * @param B        The matrix holding the load vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy_A,
                    M2 & B,
                    SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(proxy_A.lhs());
        value_type       * data_B = const_cast<value_type *>(detail::extract_raw_pointer<value_type>(B));
        
        std::size_t A_start1 = viennacl::traits::start1(proxy_A.lhs());
        std::size_t A_start2 = viennacl::traits::start2(proxy_A.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(proxy_A.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(proxy_A.lhs());
        std::size_t A_size2  = viennacl::traits::size2(proxy_A.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(proxy_A.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(proxy_A.lhs());
        
        std::size_t B_start1 = viennacl::traits::start1(B);
        std::size_t B_start2 = viennacl::traits::start2(B);
        std::size_t B_inc1   = viennacl::traits::stride1(B);
        std::size_t B_inc2   = viennacl::traits::stride2(B);
        std::size_t B_size2  = viennacl::traits::size2(B);
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(B);
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(B);
        
        
        detail::matrix_array_wrapper<value_type const, typename M1::orientation_category, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename M2::orientation_category, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        
        detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SOLVERTAG());
      }

      /** @brief Direct inplace solver for transposed triangular systems with multiple transposed right hand sides, i.e. A^T \ B^T   (MATLAB notation)
      *
      * @param proxy_A    The transposed system matrix proxy
      * @param proxy_B    The transposed matrix holding the load vectors, where the solution is directly written to
      */
      template <typename M1,
                typename M2, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_matrix<M2>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy_A,
                          matrix_expression< const M2, const M2, op_trans>   proxy_B,
                          SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<M1>::type        value_type;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(proxy_A.lhs());
        value_type       * data_B = const_cast<value_type *>(detail::extract_raw_pointer<value_type>(proxy_B.lhs()));
        
        std::size_t A_start1 = viennacl::traits::start1(proxy_A.lhs());
        std::size_t A_start2 = viennacl::traits::start2(proxy_A.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(proxy_A.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(proxy_A.lhs());
        std::size_t A_size2  = viennacl::traits::size2(proxy_A.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(proxy_A.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(proxy_A.lhs());
        
        std::size_t B_start1 = viennacl::traits::start1(proxy_B.lhs());
        std::size_t B_start2 = viennacl::traits::start2(proxy_B.lhs());
        std::size_t B_inc1   = viennacl::traits::stride1(proxy_B.lhs());
        std::size_t B_inc2   = viennacl::traits::stride2(proxy_B.lhs());
        std::size_t B_size1  = viennacl::traits::size1(proxy_B.lhs());
        std::size_t B_internal_size1  = viennacl::traits::internal_size1(proxy_B.lhs());
        std::size_t B_internal_size2  = viennacl::traits::internal_size2(proxy_B.lhs());
        
        
        detail::matrix_array_wrapper<value_type const, typename M1::orientation_category, true>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::matrix_array_wrapper<value_type,       typename M2::orientation_category, true>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
        
        detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size1, SOLVERTAG());
      }
      
      //
      //  Solve on vector
      //
      
      namespace detail
      {
        //
        // Upper solve:
        //
        template <typename MatrixType, typename VectorType>
        void upper_inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, bool unit_diagonal)
        {
          typedef typename VectorType::value_type   value_type;
          
          for (std::size_t i = 0; i < A_size; ++i)
          {
            std::size_t current_row = A_size - i - 1;
            
            for (std::size_t j = current_row + 1; j < A_size; ++j)
            {
              value_type A_element = A(current_row, j);
              b(current_row) -= A_element * b(j);
            }
            
            if (!unit_diagonal)
              b(current_row) /= A(current_row, current_row);
          }
        }
          
        template <typename MatrixType, typename VectorType>
        void inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, viennacl::linalg::unit_upper_tag)
        {
          upper_inplace_solve_vector(A, b, A_size, true);
        }
          
        template <typename MatrixType, typename VectorType>
        void inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, viennacl::linalg::upper_tag)
        {
          upper_inplace_solve_vector(A, b, A_size, false);
        }
          
        //
        // Lower solve:
        //
        template <typename MatrixType, typename VectorType>
        void lower_inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, bool unit_diagonal)
        {
          typedef typename VectorType::value_type   value_type;
          
          for (std::size_t i = 0; i < A_size; ++i)
          {
            for (std::size_t j = 0; j < i; ++j)
            {
              value_type A_element = A(i, j);
              b(i) -= A_element * b(j);
            }
            
            if (!unit_diagonal)
              b(i) /= A(i, i);
          }
        }
          
        template <typename MatrixType, typename VectorType>
        void inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, viennacl::linalg::unit_lower_tag)
        {
          lower_inplace_solve_vector(A, b, A_size, true);
        }
          
        template <typename MatrixType, typename VectorType>
        void inplace_solve_vector(MatrixType & A, VectorType & b, std::size_t A_size, viennacl::linalg::lower_tag)
        {
          lower_inplace_solve_vector(A, b, A_size, false);
        }
          
      }

      template <typename M1,
                typename V1, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  >::type
      inplace_solve(const M1 & mat,
                          V1 & vec,
                    SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type  value_type;
        typedef typename viennacl::result_of::orientation_functor<M1>::type F;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(mat);
        value_type       * data_v = detail::extract_raw_pointer<value_type>(vec);
        
        std::size_t A_start1 = viennacl::traits::start1(mat);
        std::size_t A_start2 = viennacl::traits::start2(mat);
        std::size_t A_inc1   = viennacl::traits::stride1(mat);
        std::size_t A_inc2   = viennacl::traits::stride2(mat);
        std::size_t A_size2  = viennacl::traits::size2(mat);
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(mat);
        
        std::size_t start1 = viennacl::traits::start(vec);
        std::size_t inc1   = viennacl::traits::stride(vec);
        
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::vector_array_wrapper<value_type> wrapper_v(data_v, start1, inc1);
        
        detail::inplace_solve_vector(wrapper_A, wrapper_v, A_size2, SOLVERTAG());
      }
      
      

      /** @brief Direct inplace solver for dense upper triangular systems that stem from transposed lower triangular systems
      *
      * @param proxy    The system matrix proxy
      * @param vec    The load vector, where the solution is directly written to
      */
      template <typename M1,
                typename V1, typename SOLVERTAG>
      typename viennacl::enable_if<    viennacl::is_any_dense_nonstructured_matrix<M1>::value
                                    && viennacl::is_any_dense_nonstructured_vector<V1>::value
                                  >::type
      inplace_solve(const matrix_expression< const M1, const M1, op_trans> & proxy,
                    V1 & vec,
                    SOLVERTAG)
      {
        typedef typename viennacl::result_of::cpu_value_type<V1>::type  value_type;
        typedef typename viennacl::result_of::orientation_functor<M1>::type F;
       
        value_type const * data_A = detail::extract_raw_pointer<value_type>(proxy.lhs());
        value_type       * data_v = detail::extract_raw_pointer<value_type>(vec);
        
        std::size_t A_start1 = viennacl::traits::start1(proxy.lhs());
        std::size_t A_start2 = viennacl::traits::start2(proxy.lhs());
        std::size_t A_inc1   = viennacl::traits::stride1(proxy.lhs());
        std::size_t A_inc2   = viennacl::traits::stride2(proxy.lhs());
        std::size_t A_size2  = viennacl::traits::size2(proxy.lhs());
        std::size_t A_internal_size1  = viennacl::traits::internal_size1(proxy.lhs());
        std::size_t A_internal_size2  = viennacl::traits::internal_size2(proxy.lhs());
        
        std::size_t start1 = viennacl::traits::start(vec);
        std::size_t inc1   = viennacl::traits::stride(vec);
        
        detail::matrix_array_wrapper<value_type const, typename F::orientation_category, true>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::vector_array_wrapper<value_type> wrapper_v(data_v, start1, inc1);
        
        detail::inplace_solve_vector(wrapper_A, wrapper_v, A_size2, SOLVERTAG());
      }
      
      
      
    }
  }
}

#endif
