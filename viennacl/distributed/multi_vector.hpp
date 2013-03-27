#ifndef VIENNACL_DISTRIBUTED_MULTI_VECTOR_HPP_
#define VIENNACL_DISTRIBUTED_MULTI_VECTOR_HPP_

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

/** @file multi_vector.hpp
    @brief Implementation of the distributed vector class
*/

#include "viennacl/meta/result_of.hpp"
#include "viennacl/tools/tools.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/forwards.h"

#include "viennacl/distributed/multi_matrix.hpp"

namespace viennacl{

namespace distributed{

/** @brief A dense matrix class - Multiple devices
*
* @tparam SCALARTYPE   The underlying scalar type (either float or double)
* @tparam F            Storage layout: Either row_major or column_major (at present only row_major is supported)
* @tparam ALIGNMENT   The internal memory size is given by (size()/ALIGNMENT + 1) * ALIGNMENT. ALIGNMENT must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
*/
template <class SCALARTYPE, unsigned int ALIGNMENT>
class multi_vector
{
private:
  typedef multi_vector<SCALARTYPE,ALIGNMENT> self_type;

public:
  typedef viennacl::vector<SCALARTYPE, ALIGNMENT> gpu_vector_type;
  typedef typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType   value_type;
  typedef vcl_size_t                                                   size_type;

private:
  class block_t{
      friend class multi_vector<SCALARTYPE,ALIGNMENT>;
  public:
      block_t(size_type offset, size_type size) : offset_(offset), vector_(size) { }

      SCALARTYPE & operator()(size_type index){
          return vector_(index);
      }

      SCALARTYPE operator()(size_type index) const {
          return vector_(index);
      }

  private:
      size_type offset_;
      typename viennacl::distributed::utils::gpu_wrapper<gpu_vector_type>::cpu_t vector_;
  };

  void init_blocks(){
      for(unsigned int i=0 ; i<num_blocks() ; ++i){
          size_type offset = i*block_size_;
          size_type size = std::min(block_size_, size_ - offset);
          blocks_.push_back(block_t(size,offset));
      }
  }

private:
  typedef std::vector<block_t>  blocks_t;

public:

  /** @brief Creates the matrix with the given dimensions
  *
  * @param rows     Number of rows
  * @param columns  Number of columns
  */
  explicit multi_vector(size_type size) :
      size_(size), block_size_(viennacl::distributed::utils::vector_block_size<SCALARTYPE>())
  {
    init_blocks();
  }

  size_t num_blocks() const{
      return 1 + (size_-1)/block_size_;
  }

  //read-write access to an element of the matrix
  /** @brief Read-write access to a single element of the matrix
  */
  SCALARTYPE & operator()(size_type index)
  {
      return blocks_.at(index/block_size_)(index);
  }

  /** @brief Read access to a single element of the matrix
  */
  SCALARTYPE operator()(size_type index) const
  {
      return blocks_.at(index/block_size_)(index);
  }


  /** @brief Matrix-Vector product */
  template <typename F, unsigned int MAT_ALIGNMENT>
  self_type & operator=(const vector_expression< const viennacl::distributed::multi_matrix<SCALARTYPE, F, MAT_ALIGNMENT>,
                                            const self_type,
                                            op_prod> & proxy){
      typedef const typename viennacl::distributed::multi_matrix<SCALARTYPE, F, MAT_ALIGNMENT>::gpu_matrix_type gpu_matrix_type;
      typedef gpu_vector_type& (gpu_vector_type::*FuncPtrT) (vector_expression<gpu_matrix_type, const gpu_vector_type, op_prod> const &);
      for(unsigned int row = 0 ; row < num_blocks() ; ++row){
          for(unsigned int col = 0 ; col < proxy.rhs().num_blocks() ; ++col){

              //First product is not inplace
              viennacl::distributed::task * t1 = scheduler::create_task((FuncPtrT)(&gpu_vector_type::operator=),
                                                                        viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type>(proxy.lhs().block_matrix(row,col)),
                                                                        viennacl::distributed::utils::gpu_wrapper<const gpu_vector_type>(proxy.rhs().blocks_[col].vector_),
                                                                        viennacl::distributed::utils::gpu_wrapper<gpu_vector_type>(blocks_[row].vector_)
                                                                        );
              t1->info("Matrix-Vector Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : Initial assignment " );

              //Inplace add of products
//              for(unsigned int update = 1 ; update < proxy.lhs().num_blocks_columns() ; ++update){
//                  viennacl::distributed::task * t2 = scheduler::create_task((FuncPtrT)(&gpu_matrix_type::operator+=),
//                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type1>(proxy.lhs().blocks_[row][update].matrix_),
//                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type2>(proxy.rhs().blocks_[update][col].matrix_),
//                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_));
//                  t2->info("Matrix Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : " + viennacl::tools::to_string(update));
//                  scheduler::connect(t1,t2);
//                  t1 = t2;
//              }

          }
      }
      scheduler::init();
      return *this;
  }

private:
  size_type size_;
  size_type block_size_;
  blocks_t blocks_;
};

}

}

#endif
