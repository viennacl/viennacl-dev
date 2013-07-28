#ifndef VIENNACL_SCHEDULER_EXECUTE_VECTOR_DISPATCHER_HPP
#define VIENNACL_SCHEDULER_EXECUTE_VECTOR_DISPATCHER_HPP

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


/** @file viennacl/scheduler/execute_vector_dispatcher.hpp
    @brief Provides wrappers for av(), avbv(), avbv_v(), etc. in viennacl/linalg/vector_operations.hpp such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/linalg/vector_operations.hpp"

namespace viennacl
{
  namespace scheduler
  {
    namespace detail
    {
      // helper routines for extracting the scalar type
      inline float convert_to_float(float f) { return f; }
      inline float convert_to_float(lhs_rhs_element_2 const & el)
      {
        if (el.type == HOST_SCALAR_FLOAT_TYPE)
          return el.data.host_float;
        if (el.type == SCALAR_FLOAT_TYPE)
          return *el.data.scalar_float;

        throw statement_not_supported_exception("Cannot convert to float");
      }

      // helper routines for extracting the scalar type
      inline double convert_to_double(double d) { return d; }
      inline double convert_to_double(lhs_rhs_element_2 const & el)
      {
        if (el.type == HOST_SCALAR_DOUBLE_TYPE)
          return el.data.host_double;
        if (el.type == SCALAR_DOUBLE_TYPE)
          return *el.data.scalar_double;

        throw statement_not_supported_exception("Cannot convert to double");
      }

      /** @brief Wrapper for viennacl::linalg::av(), taking care of the argument unwrapping */
      template <typename ScalarType1>
      void av(lhs_rhs_element_2 & vec1,
              lhs_rhs_element_2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
      {
        assert(vec1.type_family == VECTOR_TYPE_FAMILY && vec2.type_family == VECTOR_TYPE_FAMILY && bool("Arguments are not vector types!"));

        switch (vec1.type)
        {
          case VECTOR_FLOAT_TYPE:
            assert(vec2.type == VECTOR_FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::av(*vec1.data.vector_float,
                                 *vec2.data.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
            break;
          case VECTOR_DOUBLE_TYPE:
            assert(vec2.type == VECTOR_DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::av(*vec1.data.vector_double,
                                 *vec2.data.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
            break;
          default:
            throw statement_not_supported_exception("Invalid arguments in scheduler when calling av()");
        }
      }

      /** @brief Wrapper for viennacl::linalg::avbv(), taking care of the argument unwrapping */
      template <typename ScalarType1, typename ScalarType2>
      void avbv(lhs_rhs_element_2 & vec1,
                lhs_rhs_element_2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                lhs_rhs_element_2 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(vec1.type_family == VECTOR_TYPE_FAMILY && vec2.type_family == VECTOR_TYPE_FAMILY && bool("Arguments are not vector types!"));

        switch (vec1.type)
        {
          case VECTOR_FLOAT_TYPE:
            assert(vec2.type == VECTOR_FLOAT_TYPE && vec3.type == VECTOR_FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::avbv(*vec1.data.vector_float,
                                   *vec2.data.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                                   *vec3.data.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
            break;
          case VECTOR_DOUBLE_TYPE:
            assert(vec2.type == VECTOR_DOUBLE_TYPE && vec3.type == VECTOR_DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::avbv(*vec1.data.vector_double,
                                   *vec2.data.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                                   *vec3.data.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
            break;
          default:
            throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv()");
        }
      }

      /** @brief Wrapper for viennacl::linalg::avbv_v(), taking care of the argument unwrapping */
      template <typename ScalarType1, typename ScalarType2>
      void avbv_v(lhs_rhs_element_2 & vec1,
                  lhs_rhs_element_2 const & vec2, ScalarType1 const & alpha, std::size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                  lhs_rhs_element_2 const & vec3, ScalarType2 const & beta,  std::size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
      {
        assert(vec1.type_family == VECTOR_TYPE_FAMILY && vec2.type_family == VECTOR_TYPE_FAMILY && bool("Arguments are not vector types!"));

        switch (vec1.type)
        {
          case VECTOR_FLOAT_TYPE:
            assert(vec2.type == VECTOR_FLOAT_TYPE && vec3.type == VECTOR_FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::avbv_v(*vec1.data.vector_float,
                                     *vec2.data.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                                     *vec3.data.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
            break;
          case VECTOR_DOUBLE_TYPE:
            assert(vec2.type == VECTOR_DOUBLE_TYPE && vec3.type == VECTOR_DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
            viennacl::linalg::avbv_v(*vec1.data.vector_double,
                                     *vec2.data.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                                     *vec3.data.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
            break;
          default:
            throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv_v()");
        }
      }


    } // namespace detail
  } // namespace scheduler
} // namespace viennacl

#endif

