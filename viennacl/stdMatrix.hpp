#ifndef VIENNACL_STDMATRIX_HPP_
#define VIENNACL_STDMATRIX_HPP_

/* =========================================================================
Copyright (c) 2010-2015, Institute for Microelectronics,
Institute for Analysis and Scientific Computing,
TU Wien.
Portions of this software are copyright by UChicago Argonne, LLC.

-----------------
ViennaCL - The Vienna Computing Library
-----------------

Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

(A list of authors and contributors can be found in the manual)

License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/stdMatrix.hpp
@brief A very primitive implementation of the CPU matrix class based on std::vector
*/

#include "viennacl/tools/tools.hpp"
#include "viennacl/traits/row_major.hpp"

namespace viennacl
{
  template <typename NumericT, typename F = viennacl::row_major> class StdMatrix: public std::vector<NumericT>
  {
  public:
    typedef typename std::vector<NumericT> vecType;
    typedef typename vecType::size_type size_type;

    size_type size2_, size1_;
    size_type internal_size1_, internal_size2_;
    StdMatrix(): vecType() {};
    StdMatrix(size_type size1, size_type size2):StdMatrix()
    {
      resize(size1, size2);
    };
    StdMatrix(size_type rows, size_type columns, vecType &v) : StdMatrix(rows, columns) {
      copy(v, (*this));
    };

    using vecType::resize;
    void resize(size_type size1, size_type size2)
    {
      size1_ = size1;
      size2_ = size2;
      internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
      internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
      reserve(must_be_internal_size());
      resize(must_be_internal_size());
    }
    /** @brief Returns the internal number of rows. Usually required for launching OpenCL kernels only */
    size_type internal_size1() const { return internal_size1_; }
    /** @brief Returns the internal number of columns. Usually required for launching OpenCL kernels only */
    size_type internal_size2() const { return internal_size2_; }
    /** @brief Returns the total amount of allocated memory in multiples of sizeof(NumericT) */
    size_type internal_size() const { return vecType::size(); }

    size_type must_be_internal_size() const { return internal_size1()*internal_size2(); }

    /** @brief Returns the number of rows */
    size_type size1() const { return size1_; }
    /** @brief Returns the number of columns */
    size_type size2() const { return size2_; }
    /** @brief Returns the total count of elements */
    size_type size() const { return  size1()* size2(); }


    class entry_proxy{
      NumericT &ptr;
    public:
      entry_proxy(NumericT &ptr):ptr(ptr) {};
      NumericT operator=(NumericT &a)
      {
        ptr = a;
        return a;
      };
    };
    entry_proxy operator()(size_type i, size_type j)
    {
      return entry_proxy((*this)[F::mem_index(i, j, internal_size1(), internal_size2())]);
    }
    /*vecType operator[](size_type i)
    {
      auto b=begin()+F::mem_index(i, 0, internal_size1(), internal_size2())];
      auto e = b + internal_size1();
      return vecType(b,e);
    }*/
  };



  /** @brief Copies a dense StdMatrix<> to std::vector<>
  *
  * @param vector_matrix   A dense matrix on the host of type std::vector<>. cpu_matrix[F::mem_index(i, j, gpu_matrix.size1(), gpu_matrix.size2())] returns the element in the i-th row and j-th columns (both starting with zero)
  * @param std_matrix      A dense matrix on the host of type StdMatrix<>.
  */
  template<typename NumericT, typename A, typename F>
  void copy(std::vector<NumericT, A> & vector_matrix, StdMatrix<NumericT, F> & std_matrix)
  {
    typedef typename matrix<NumericT, F>::size_type      size_type;
    if (std_matrix.internal_size() == vector_matrix.size())
    {
      std::copy(begin(vector_matrix), end(vector_matrix), begin(std_matrix));
    }
    else if (std_matrix.size() == vector_matrix.size())
    {
      std::vector<NumericT> data(std_matrix.internal_size());
      for (size_type i = 0; i < std_matrix.size1(); ++i)
      {
        for (size_type j = 0; j < std_matrix.size2(); ++j)
          //std_matrix[F::mem_index(i, j, std_matrix.internal_size1(), std_matrix.internal_size2())] = vector_matrix[F::mem_index(i, j, std_matrix.size1(), std_matrix.size2())];
          std_matrix(i, j) = vector_matrix[F::mem_index(i, j, std_matrix.size1(), std_matrix.size2())];
      }
    }
    else
      assert(false && bool("Matrix dimensions mismatch."));
  }

  /** @brief Copies a dense StdMatrix<> from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
  *
  * @param cpu_matrix   A dense matrix on the host of type StdMatrix<>.
  * @param gpu_matrix   A dense ViennaCL matrix
  */
  template<typename NumericT, typename F, unsigned int AlignmentV>
  void copy(StdMatrix<NumericT, F> & cpu_matrix,
    matrix<NumericT, F, AlignmentV> & gpu_matrix)
  {
    typedef typename matrix<NumericT, F, AlignmentV>::size_type      size_type;

    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.size1(), cpu_matrix.size2(), false);
    }
    assert(gpu_matrix.size1() == cpu_matrix.size1() && bool("Matrix size1 mismatch."));
    assert(gpu_matrix.size2() == cpu_matrix.size2() && bool("Matrix size2 mismatch."));
    if (gpu_matrix.internal_size() == cpu_matrix.internal_size()) {
      assert(gpu_matrix.internal_size1() == cpu_matrix.internal_size1() && bool("Matrix internal size1 mismatch."));
      assert(gpu_matrix.internal_size2() == cpu_matrix.internal_size2() && bool("Matrix internal size2 mismatch."));
      viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * cpu_matrix.internal_size(), &(cpu_matrix[0]));
    }
    else {
      std::vector<NumericT> data(gpu_matrix.internal_size());
      for (size_type i = 0; i < gpu_matrix.size1(); ++i)
      {
        for (size_type j = 0; j < gpu_matrix.size2(); ++j)
          data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix[F::mem_index(i, j, gpu_matrix.size1(), gpu_matrix.size2())];
      }

      viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * data.size(), &(data[0]));
    }
    //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  }

};
#endif
