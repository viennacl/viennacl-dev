#ifndef VIENNACL_META_PREDICATE_HPP_
#define VIENNACL_META_PREDICATE_HPP_

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

/** @file predicate.hpp
    @brief All the predicates used within ViennaCL. Checks for expressions to be vectors, etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

namespace viennacl
{
    /** \cond */  //deactivate Doxygen parsing of the partial specializations

    //
    // is_cpu_scalar: checks for float or double
    //
    //template <typename T>
    //struct is_cpu_scalar
    //{
    //  enum { value = false };
    //};
  
    template <>
    struct is_cpu_scalar<float>
    {
      enum { value = true };
    };

    template <>
    struct is_cpu_scalar<double>
    {
      enum { value = true };
    };
    
    //
    // is_scalar: checks for viennacl::scalar
    //
    //template <typename T>
    //struct is_scalar
    //{
    //  enum { value = false };
    //};
  
    template <typename T>
    struct is_scalar<viennacl::scalar<T> >
    {
      enum { value = true };
    };

    //
    // is_flip_sign_scalar: checks for viennacl::scalar modified with unary operator-
    //
    //template <typename T>
    //struct is_flip_sign_scalar
    //{
    //  enum { value = false };
    //};
  
    template <typename T>
    struct is_flip_sign_scalar<viennacl::scalar_expression< const scalar<T>,
                                                            const scalar<T>,
                                                            op_flip_sign> >
    {
      enum { value = true };
    };
    
    //
    // is_any_scalar: checks for either CPU and GPU scalars, i.e. is_cpu_scalar<>::value || is_scalar<>::value
    //
    //template <typename T>
    //struct is_any_scalar
    //{
    //  enum { value = (is_scalar<T>::value || is_cpu_scalar<T>::value || is_flip_sign_scalar<T>::value )};
    //};

    //
    // is_row_major
    //
    //template <typename T>
    //struct is_row_major
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType>
    struct is_row_major<viennacl::matrix_base<ScalarType, viennacl::row_major> >
    {
      enum { value = true };
    };

    template <>
    struct is_row_major< viennacl::row_major >
    {
      enum { value = true };
    };

    template <typename T>
    struct is_row_major<viennacl::matrix_expression<T, T, viennacl::op_trans> >
    {
      enum { value = is_row_major<T>::value };
    };
    
    
    //
    // is_circulant_matrix
    //
    //template <typename T>
    //struct is_circulant_matrix
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_circulant_matrix<viennacl::circulant_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };
    
    //
    // is_hankel_matrix
    //
    //template <typename T>
    //struct is_hankel_matrix
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_hankel_matrix<viennacl::hankel_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };
    
    //
    // is_toeplitz_matrix
    //
    //template <typename T>
    //struct is_toeplitz_matrix
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_toeplitz_matrix<viennacl::toeplitz_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };
    
    //
    // is_vandermonde_matrix
    //
    //template <typename T>
    //struct is_vandermonde_matrix
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_vandermonde_matrix<viennacl::vandermonde_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };
    
    
    
    //
    // is_compressed_matrix
    //
    template <typename T>
    struct is_compressed_matrix
    {
      enum { value = false };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_compressed_matrix<viennacl::compressed_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    //
    // is_coordinate_matrix
    //
    template <typename T>
    struct is_coordinate_matrix
    {
      enum { value = false };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_coordinate_matrix<viennacl::coordinate_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    //
    // is_ell_matrix
    //
    template <typename T>
    struct is_ell_matrix
    {
      enum { value = false };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_ell_matrix<viennacl::ell_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    //
    // is_hyb_matrix
    //
    template <typename T>
    struct is_hyb_matrix
    {
      enum { value = false };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_hyb_matrix<viennacl::hyb_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    
    //
    // is_any_sparse_matrix
    //
    //template <typename T>
    //struct is_any_sparse_matrix
    //{
    //  enum { value = false };
    //};

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_any_sparse_matrix<viennacl::compressed_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_any_sparse_matrix<viennacl::coordinate_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_any_sparse_matrix<viennacl::ell_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    template <typename ScalarType, unsigned int ALIGNMENT>
    struct is_any_sparse_matrix<viennacl::hyb_matrix<ScalarType, ALIGNMENT> >
    {
      enum { value = true };
    };

    /** \endcond */
    
    //////////////// Part 2: Operator predicates ////////////////////
    
    //
    // is_addition
    //
    /** @brief Helper metafunction for checking whether the provided type is viennacl::op_add (for addition) */
    template <typename T>
    struct is_addition
    {
      enum { value = false };
    };

    /** \cond */
    template <>
    struct is_addition<viennacl::op_add>
    {
      enum { value = true };
    };
    /** \endcond */
    
    //
    // is_subtraction
    //
    /** @brief Helper metafunction for checking whether the provided type is viennacl::op_sub (for subtraction) */
    template <typename T>
    struct is_subtraction
    {
      enum { value = false };
    };
    
    /** \cond */
    template <>
    struct is_subtraction<viennacl::op_sub>
    {
      enum { value = true };
    };
    /** \endcond */
    
    //
    // is_product
    //
    /** @brief Helper metafunction for checking whether the provided type is viennacl::op_prod (for products/multiplication) */
    template <typename T>
    struct is_product
    {
      enum { value = false };
    };

    /** \cond */
    template <>
    struct is_product<viennacl::op_prod>
    {
      enum { value = true };
    };
    /** \endcond */
    
    //
    // is_division
    //
    /** @brief Helper metafunction for checking whether the provided type is viennacl::op_div (for division) */
    template <typename T>
    struct is_division
    {
      enum { value = false };
    };

    /** \cond */
    template <>
    struct is_division<viennacl::op_div>
    {
      enum { value = true };
    };
    /** \endcond */
    
    
} //namespace viennacl
    

#endif
