#ifndef VIENNACL_TOOLS_TOOLS_HPP_
#define VIENNACL_TOOLS_TOOLS_HPP_

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

/** @file viennacl/tools/tools.hpp
    @brief Various little tools used here and there in ViennaCL.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/tools/adapter.hpp"

#include <vector>
#include <map>

namespace viennacl
{
  namespace tools
  {
    
    /** @brief Supply suitable increment functions for the iterators: */
    template <class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    struct MATRIX_ITERATOR_INCREMENTER<viennacl::row_iteration, viennacl::matrix<SCALARTYPE, F, ALIGNMENT> >
    {
      static void apply(const viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & mat, unsigned int & row, unsigned int & col) { ++row; }
    };

    template <class SCALARTYPE, typename F, unsigned int ALIGNMENT>
    struct MATRIX_ITERATOR_INCREMENTER<viennacl::col_iteration, viennacl::matrix<SCALARTYPE, F, ALIGNMENT> >
    {
      static void apply(const viennacl::matrix<SCALARTYPE, F, ALIGNMENT> & mat, unsigned int & row, unsigned int & col) { ++col; }
    };

    
    /** @brief A guard that checks whether the floating point type of GPU types is either float or double */
    template <typename T>
    struct CHECK_SCALAR_TEMPLATE_ARGUMENT
    {
        typedef typename T::ERROR_SCALAR_MUST_HAVE_TEMPLATE_ARGUMENT_FLOAT_OR_DOUBLE  ResultType;
    };
    
    template <>
    struct CHECK_SCALAR_TEMPLATE_ARGUMENT<float>
    {
        typedef float  ResultType;
    };
    
    template <>
    struct CHECK_SCALAR_TEMPLATE_ARGUMENT<double>
    {
        typedef double  ResultType;
    };

    
    
    /** @brief Reads a text from a file into a std::string
    *
    * @param filename   The filename
    * @return The text read from the file
    */
    inline std::string readTextFromFile(const std::string & filename)
    {
      std::ifstream f(filename.c_str());
      if (!f) return std::string();

      std::stringstream result;
      std::string tmp;
      while (std::getline(f, tmp))
        result << tmp << std::endl;

      return result.str();
    }

    /** @brief Replaces all occurances of a substring by another stringstream
    *
    * @param text   The string to search in
    * @param to_search  The substring to search for
    * @param to_replace The replacement for found substrings
    * @return The resulting string
    */
    inline std::string strReplace(const std::string & text, std::string to_search, std::string to_replace)
    {
      std::string::size_type pos = 0;
      std::string result;
      std::string::size_type found;
      while( (found = text.find(to_search, pos)) != std::string::npos )
      {
        result.append(text.substr(pos,found-pos));
        result.append(to_replace);
        pos = found + to_search.length();
      }
      if (pos < text.length())
        result.append(text.substr(pos));
      return result;
    }

    /** @brief Rounds an integer to the next multiple of another integer
    *
    * @tparam INT_TYPE  The integer type
    * @param to_reach   The integer to be rounded up (ceil operation)
    * @param base       The base
    * @return The smallest multiple of 'base' such that to_reach <= base
    */
    template <class INT_TYPE>
    INT_TYPE roundUpToNextMultiple(INT_TYPE to_reach, INT_TYPE base)
    {
      if (to_reach % base == 0) return to_reach;
      return ((to_reach / base) + 1) * base;
    }
    
    
    /** @brief Create a double precision kernel out of a single precision kernel
    *
    * @param source          The source string
    * @param fp_extension    An info string that specifies the OpenCL double precision extension
    * @return   The double precision kernel
    */
    inline std::string make_double_kernel(std::string const & source, std::string const & fp_extension)
    {
      std::stringstream ss;
      ss << "#pragma OPENCL EXTENSION " << fp_extension << " : enable\n\n";
      
      std::string result = ss.str();
      result.append(strReplace(source, "float", "double"));
      return result;
    }
    
    
    /** @brief Removes the const qualifier from a type */
    template <typename T>
    struct CONST_REMOVER
    {
      typedef T   ResultType;
    };

    template <typename T>
    struct CONST_REMOVER<const T>
    {
      typedef T   ResultType;
    };


    /** @brief Extracts the vector type from one of the two arguments. Used for the vector_expression type.
    *
    * @tparam LHS   The left hand side operand of the vector_expression
    * @tparam RHS   The right hand side operand of the vector_expression
    */
    template <typename LHS, typename RHS>
    struct VECTOR_EXTRACTOR_IMPL
    {
      typedef typename LHS::ERROR_COULD_NOT_EXTRACT_VECTOR_INFORMATION_FROM_VECTOR_EXPRESSION  ResultType;
    };
    
    template <typename LHS, typename ScalarType, unsigned int A>
    struct VECTOR_EXTRACTOR_IMPL<LHS, viennacl::vector<ScalarType, A> >
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };

    template <typename LHS, typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<LHS, viennacl::vector_range<VectorType> >
    {
      typedef VectorType   ResultType;
    };

    template <typename LHS, typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<LHS, viennacl::vector_slice<VectorType> >
    {
      typedef VectorType   ResultType;
    };

    
    template <typename RHS, typename ScalarType, unsigned int A>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A>, RHS>
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };

    template <typename VectorType, typename RHS>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_range<VectorType>, RHS>
    {
      typedef VectorType   ResultType;
    };

    template <typename VectorType, typename RHS>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_slice<VectorType>, RHS>
    {
      typedef VectorType   ResultType;
    };

    template <typename ScalarType, unsigned int A1, unsigned int A2>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A1>, viennacl::vector<ScalarType, A2> >
    {
      typedef viennacl::vector<ScalarType, A1>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A>, viennacl::vector_range<VectorType> >
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A>, viennacl::vector_slice<VectorType> >
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };
    
    
    template <typename VectorType, typename ScalarType, unsigned int A>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_range<VectorType>, viennacl::vector<ScalarType, A> >
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };
    
    template <typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_range<VectorType>, viennacl::vector_range<VectorType> >
    {
      typedef VectorType   ResultType;
    };

    template <typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_range<VectorType>, viennacl::vector_slice<VectorType> >
    {
      typedef VectorType   ResultType;
    };

    
    template <typename VectorType, typename ScalarType, unsigned int A>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_slice<VectorType>, viennacl::vector<ScalarType, A> >
    {
      typedef viennacl::vector<ScalarType, A>   ResultType;
    };
    
    template <typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_slice<VectorType>, viennacl::vector_range<VectorType> >
    {
      typedef VectorType   ResultType;
    };
    
    template <typename VectorType>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_slice<VectorType>, viennacl::vector_slice<VectorType> >
    {
      typedef VectorType   ResultType;
    };
    
    
    // adding vector_expression to the resolution:
    template <typename LHS, typename RHS>
    struct VECTOR_EXTRACTOR;

    template <typename LHS, typename V2, typename S2, typename OP2>
    struct VECTOR_EXTRACTOR_IMPL<LHS, viennacl::vector_expression<const V2, const S2, OP2> >
    {
      typedef typename VECTOR_EXTRACTOR<V2, S2>::ResultType      ResultType;
    };
    
    //resolve ambiguities for previous cases:
    template <typename ScalarType, unsigned int A, typename V2, typename S2, typename OP2>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A>, viennacl::vector_expression<const V2, const S2, OP2> >
    {
      typedef viennacl::vector<ScalarType, A>      ResultType;
    };

    template <typename VectorType, typename V2, typename S2, typename OP2>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_range<VectorType>, viennacl::vector_expression<const V2, const S2, OP2> >
    {
      typedef VectorType   ResultType;
    };

    template <typename VectorType, typename V2, typename S2, typename OP2>
    struct VECTOR_EXTRACTOR_IMPL<viennacl::vector_slice<VectorType>, viennacl::vector_expression<const V2, const S2, OP2> >
    {
      typedef VectorType   ResultType;
    };

    
    
    
    template <typename LHS, typename RHS>
    struct VECTOR_EXTRACTOR
    {
      typedef typename VECTOR_EXTRACTOR_IMPL<typename CONST_REMOVER<LHS>::ResultType,
                                              typename CONST_REMOVER<RHS>::ResultType>::ResultType      ResultType;
    };

    /////// Same for matrices: matrix_extractor ///////////////
    
    
    /** @brief Extracts the vector type from one of the two arguments. Used for the vector_expression type.
    *
    * @tparam LHS   The left hand side operand of the vector_expression
    * @tparam RHS   The right hand side operand of the vector_expression
    */
    template <typename LHS, typename RHS>
    struct MATRIX_EXTRACTOR_IMPL
    {
      typedef typename LHS::ERROR_COULD_NOT_EXTRACT_MATRIX_INFORMATION_FROM_MATRIX_EXPRESSION  ResultType;
    };
    
    template <typename LHS, typename ScalarType, typename F, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<LHS, viennacl::matrix<ScalarType, F, A> >
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };

    template <typename LHS, typename ScalarType, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<LHS, viennacl::compressed_matrix<ScalarType, A> >
    {
      typedef viennacl::compressed_matrix<ScalarType, A>   ResultType;
    };
    
    template <typename LHS, typename MatrixType>
    struct MATRIX_EXTRACTOR_IMPL<LHS, viennacl::matrix_range<MatrixType> >
    {
      typedef MatrixType   ResultType;
    };

    template <typename LHS, typename MatrixType>
    struct MATRIX_EXTRACTOR_IMPL<LHS, viennacl::matrix_slice<MatrixType> >
    {
      typedef MatrixType   ResultType;
    };

    
    template <typename RHS, typename ScalarType, typename F, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix<ScalarType, F, A>, RHS>
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };

    template <typename RHS, typename ScalarType, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::compressed_matrix<ScalarType, A>, RHS>
    {
      typedef viennacl::compressed_matrix<ScalarType, A>   ResultType;
    };
    
    template <typename MatrixType, typename RHS>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_range<MatrixType>, RHS>
    {
      typedef MatrixType   ResultType;
    };

    template <typename MatrixType, typename RHS>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_slice<MatrixType>, RHS>
    {
      typedef MatrixType   ResultType;
    };

    template <typename ScalarType, typename F1, typename F2, unsigned int A1, unsigned int A2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix<ScalarType, F1, A1>, viennacl::matrix<ScalarType, F2, A2> >
    {
      typedef viennacl::matrix<ScalarType, F1, A1>   ResultType;
    };

    template <typename ScalarType, unsigned int A1, unsigned int A2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::compressed_matrix<ScalarType, A1>, viennacl::compressed_matrix<ScalarType, A2> >
    {
      typedef viennacl::compressed_matrix<ScalarType, A1>   ResultType;
    };
    
    template <typename ScalarType, typename F, unsigned int A, typename MatrixType>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix<ScalarType, F, A>, viennacl::matrix_range<MatrixType> >
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };

    template <typename ScalarType, typename F, unsigned int A, typename MatrixType>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix<ScalarType, F, A>, viennacl::matrix_slice<MatrixType> >
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };
    
    
    template <typename MatrixType, typename F, typename ScalarType, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_range<MatrixType>, viennacl::matrix<ScalarType, F, A> >
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };
    
    template <typename MatrixType1, typename MatrixType2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_range<MatrixType1>, viennacl::matrix_range<MatrixType2> >
    {
      typedef MatrixType1   ResultType;
    };

    template <typename MatrixType1, typename MatrixType2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_range<MatrixType1>, viennacl::matrix_slice<MatrixType2> >
    {
      typedef MatrixType1   ResultType;
    };

    
    template <typename MatrixType, typename ScalarType, typename F, unsigned int A>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_slice<MatrixType>, viennacl::matrix<ScalarType, F, A> >
    {
      typedef viennacl::matrix<ScalarType, F, A>   ResultType;
    };
    
    template <typename MatrixType1, typename MatrixType2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_slice<MatrixType1>, viennacl::matrix_range<MatrixType2> >
    {
      typedef MatrixType1   ResultType;
    };
    
    template <typename MatrixType1, typename MatrixType2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_slice<MatrixType1>, viennacl::matrix_slice<MatrixType2> >
    {
      typedef MatrixType1   ResultType;
    };
    
    
    // adding matrix_expression to the resolution:
    template <typename LHS, typename RHS>
    struct MATRIX_EXTRACTOR;

    template <typename LHS, typename V2, typename S2, typename OP2>
    struct MATRIX_EXTRACTOR_IMPL<LHS, viennacl::matrix_expression<const V2, const S2, OP2> >
    {
      typedef typename MATRIX_EXTRACTOR<V2, S2>::ResultType      ResultType;
    };
    
    //resolve ambiguities for previous cases:
    template <typename ScalarType, typename F, unsigned int A, typename V2, typename S2, typename OP2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix<ScalarType, F, A>, viennacl::matrix_expression<const V2, const S2, OP2> >
    {
      typedef viennacl::matrix<ScalarType, F, A>      ResultType;
    };

    template <typename MatrixType, typename V2, typename S2, typename OP2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_range<MatrixType>, viennacl::matrix_expression<const V2, const S2, OP2> >
    {
      typedef MatrixType   ResultType;
    };

    template <typename MatrixType, typename V2, typename S2, typename OP2>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_slice<MatrixType>, viennacl::matrix_expression<const V2, const S2, OP2> >
    {
      typedef MatrixType   ResultType;
    };

    //special case: outer vector product
    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::vector<ScalarType, A>,
                                 T >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };
    
    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::vector_range<viennacl::vector<ScalarType, A> >,
                                 T >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::vector_range<const viennacl::vector<ScalarType, A> >,
                                 T >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::vector_slice<viennacl::vector<ScalarType, A> >,
                                 T >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::vector_slice<const viennacl::vector<ScalarType, A> >,
                                 T >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    
    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_expression<const viennacl::vector<ScalarType, A>, T, op_prod>,
                                 ScalarType >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };
    
    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_expression<const viennacl::vector_range<viennacl::vector<ScalarType, A> >, T, op_prod>,
                                 ScalarType >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_expression<const viennacl::vector_range<const viennacl::vector<ScalarType, A> >, T, op_prod>,
                                 ScalarType >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_expression<const viennacl::vector_slice<viennacl::vector<ScalarType, A> >, T, op_prod>,
                                 ScalarType >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };

    template <typename ScalarType, unsigned int A, typename T>
    struct MATRIX_EXTRACTOR_IMPL<viennacl::matrix_expression<const viennacl::vector_slice<const viennacl::vector<ScalarType, A> >, T, op_prod>,
                                 ScalarType >
    {
      typedef viennacl::matrix<ScalarType, viennacl::row_major>   ResultType;
    };
    
    
    
    template <typename LHS, typename RHS>
    struct MATRIX_EXTRACTOR
    {
      typedef typename MATRIX_EXTRACTOR_IMPL<typename CONST_REMOVER<LHS>::ResultType,
                                             typename CONST_REMOVER<RHS>::ResultType>::ResultType      ResultType;
    };
    
    
    
    
    
    
    
    
    
    /////// CPU scalar type deducer ///////////
    
    /** @brief Obtain the cpu scalar type from a type, including a GPU type like viennacl::scalar<T>
    *
    * @tparam T   Either a CPU scalar type or a GPU scalar type
    */
    template <typename T>
    struct CPU_SCALAR_TYPE_DEDUCER
    {
      //force compiler error if type cannot be deduced
      //typedef T       ResultType;
    };

    template <>
    struct CPU_SCALAR_TYPE_DEDUCER< float >
    {
      typedef float       ResultType;
    };

    template <>
    struct CPU_SCALAR_TYPE_DEDUCER< double >
    {
      typedef double       ResultType;
    };
    
    template <typename T>
    struct CPU_SCALAR_TYPE_DEDUCER< viennacl::scalar<T> >
    {
      typedef T       ResultType;
    };

    template <typename T, unsigned int A>
    struct CPU_SCALAR_TYPE_DEDUCER< viennacl::vector<T, A> >
    {
      typedef T       ResultType;
    };

    template <typename T, typename F, unsigned int A>
    struct CPU_SCALAR_TYPE_DEDUCER< viennacl::matrix<T, F, A> >
    {
      typedef T       ResultType;
    };

    
    template <typename T, typename F, unsigned int A>
    struct CPU_SCALAR_TYPE_DEDUCER< viennacl::matrix_expression<const matrix<T, F, A>, const matrix<T, F, A>, op_trans> >
    {
      typedef T       ResultType;
    };

    //
    // Converts a scalar type when necessary unless it is a viennacl::scalar<> (typical use-case: convert user-provided floats to double (and vice versa) for OpenCL kernels)
    //
    
    template <typename HostScalarType>
    viennacl::scalar<HostScalarType> const & promote_if_host_scalar(viennacl::scalar<HostScalarType> const & s) { return s; }

    template <typename HostScalarType>
    viennacl::scalar_expression<const viennacl::scalar<HostScalarType>,
                                const viennacl::scalar<HostScalarType>,
                                viennacl::op_flip_sign> const & 
    promote_if_host_scalar(viennacl::scalar_expression<const viennacl::scalar<HostScalarType>,
                                                       const viennacl::scalar<HostScalarType>,
                                                       viennacl::op_flip_sign> const & s) { return s; }
    
    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(float s) { return s; }

    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(double s) { return s; }
    
    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(long s) { return s; }
    
    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(unsigned long s) { return s; }
    
    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(int s) { return s; }
    
    template <typename HostScalarType>
    HostScalarType promote_if_host_scalar(unsigned int s) { return s; }
    
  } //namespace tools
} //namespace viennacl
    

#endif
