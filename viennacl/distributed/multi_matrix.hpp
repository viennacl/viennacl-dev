#ifndef VIENNACL_DISTRIBUTED_MULTI_MATRIX_HPP_
#define VIENNACL_DISTRIBUTED_MULTI_MATRIX_HPP_

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

/** @file multi_matrix.hpp
    @brief Implementation of the distributed dense matrix class
*/

#include "viennacl/distributed/scheduler.hpp"
#include "viennacl/distributed/forwards.hpp"
#include "viennacl/distributed/utils.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/prod.hpp"

#include "viennacl/generator/custom_operation.hpp"

#include <numeric>

namespace viennacl
{   

namespace distributed
{

//template<class Mat1, class Mat2, class Mat3>
//static void autotuned_prod_impl_1(Mat1 & mat1, Mat2 const & mat2, Mat3 const & mat3){
//    viennacl::generator::dummy_matrix<Mat1> m1(mat1);
//    viennacl::generator::dummy_matrix<Mat2> m2(mat2);
//    viennacl::generator::dummy_matrix<Mat3> m3(mat3);
//    assert(mat1.size1()==mat2.size1());
//    assert(mat1.size2()==mat3.size2());
//    assert(mat2.size2()==mat3.size1());
//    viennacl::generator::custom_operation op( m1 = viennacl::generator::prod(m2,m3));
//    op.execute();
////    mat1=viennacl::linalg::prod(mat2,mat3);
//}

//template<class Mat1, class Mat2, class Mat3>
//static void autotuned_prod_impl_2(Mat1 & mat1, Mat2 const & mat2, Mat3 const & mat3){
//    viennacl::generator::dummy_matrix<Mat1> m1(mat1);
//    viennacl::generator::dummy_matrix<Mat2> m2(mat2);
//    viennacl::generator::dummy_matrix<Mat3> m3(mat3);
//    assert(mat1.size1()==mat2.size1());
//    assert(mat1.size2()==mat3.size2());
//    assert(mat2.size2()==mat3.size1());
//    viennacl::generator::custom_operation op( m1 = viennacl::generator::prod(m2,m3));
//    op.execute();
////    mat1+=viennacl::linalg::prod(mat2,mat3);
//}


template<class T>
struct multi_to_wrapper;

template<class ScalarType, class F>
struct multi_to_wrapper<multi_matrix<ScalarType, F> >{
    typedef utils::gpu_wrapper<matrix<ScalarType,F> > type;
};

class multi_to_wrapper_fun{
public:
    multi_to_wrapper_fun(unsigned int i, unsigned int j): i_(i), j_(j){ }
    template<class T>
    typename multi_to_wrapper<T>::type operator()(T const & t) const{
        return typename multi_to_wrapper<T>::type(t.block_matrix(i_,j_));
    }

private:
    unsigned int i_;
    unsigned int j_;
};




template<class T>
struct get_first_leave{
    typedef T result_type;
    static result_type const & result(T const & t) { return t; }
};

template<class LHS, class OP, class RHS>
struct get_first_leave<generator::binary_matrix_expression_wrapper<LHS,OP,RHS> >{
    typedef typename get_first_leave<LHS>::result_type result_type;
    static result_type  const & result(generator::binary_matrix_expression_wrapper<LHS,OP,RHS> const & t){
        return get_first_leave<LHS>::result(t.lhs());
    }
};



struct alloc_fun{
    template<class T>
    void operator()(utils::gpu_wrapper<T> const & t){ t.alloc(); }
};

struct free_fun{
    template<class T>
    void operator()(utils::gpu_wrapper<T> const & t){ t.free(); }
};

template<class T>
struct wrapper_to_matrix;

template<class T>
struct wrapper_to_matrix<utils::gpu_wrapper<T> >{
    typedef generator::dummy_matrix<T> type;
};

class wrapper_to_matrix_fun{
public:
    template<class T>
    generator::dummy_matrix<T> operator()(utils::gpu_wrapper<T> const & t) const{
        return generator::dummy_matrix<T>(*t.gpu_structure_ptr());
    }
};

//    typedef typename utils::transform<A,wrapper_to_matrix,wrapper_to_matrix_fun>::result_type Tmp2Type;
//    typedef typename utils::transform<A,wrapper_to_matrix,wrapper_to_matrix_fun>::result_type Tmp2Type;
//    Tmp2Type tmp2(utils::transform<A,wrapper_to_matrix,wrapper_to_matrix_fun>::result(args_,wrapper_to_matrix_fun()));
//    fun_(args_.rhs(),utils::make_shallow_copy<Tmp2Type>::result(tmp2));



//    typedef typename T::Rhs::Lhs MatType1;
//    typedef typename T::Rhs::Rhs MatType2;
//    typedef typename T::Lhs Res;
//    typedef typename get_first_leave<MatType1>::result_type FirstLhsT;
//    typedef typename get_first_leave<MatType2>::result_type FirstRhsT;

template<class A>
static void enqueue_op(A const & wrappers){
    utils::execute<A>()(wrappers,alloc_fun());
    typedef typename utils::transform<A,wrapper_to_matrix,wrapper_to_matrix_fun>::result_type TmpType;
    TmpType tmp(utils::transform<A,wrapper_to_matrix,wrapper_to_matrix_fun>::result(wrappers,wrapper_to_matrix_fun()));
    viennacl::generator::custom_operation op(utils::make_shallow_copy<TmpType>::result(tmp));
    op.execute();
    viennacl::backend::finish();
    wrappers.lhs().transfer_back();
    utils::execute<A>()(wrappers,free_fun());
}

template<class A>
static void enqueue_op_prod(A const & wrappers){

    typedef typename A::Lhs ResT;
    typedef typename A::Rhs ProdT;
    typedef typename ProdT::Lhs ProdLhs;
    typedef typename ProdT::Rhs ProdRhs;



    //Wrapping LHS
    utils::execute<ProdLhs>()(wrappers.rhs().lhs(),alloc_fun());
    typedef utils::transform<ProdLhs,wrapper_to_matrix,wrapper_to_matrix_fun> TmpTransformLhsT;
    typename TmpTransformLhsT::result_type tmplhs(TmpTransformLhsT::result(wrappers.rhs().lhs(),wrapper_to_matrix_fun()));
    typedef typename utils::make_shallow_copy<typename TmpTransformLhsT::result_type>::result_type MatType1;
    MatType1 real_lhs(utils::make_shallow_copy<typename TmpTransformLhsT::result_type>::result(tmplhs));
    typedef typename get_first_leave<MatType1>::result_type FirstLhsT;
    FirstLhsT first_lhs = get_first_leave<MatType1>::result(real_lhs);
    typename FirstLhsT::gpu_type * lhs_ptr = NULL;
    bool make_lhs_tmp = utils::n_handles(real_lhs)>1;

    if(make_lhs_tmp){
//        std::cout << "Creating LHS Temporary..." << std::endl;
        lhs_ptr = new typename FirstLhsT::gpu_type(first_lhs.mat().size1(), first_lhs.mat().size2());
        viennacl::generator::custom_operation op;
        op.add(FirstLhsT(*lhs_ptr) = real_lhs);
        op.execute();
        viennacl::backend::finish();
        utils::execute<ProdLhs>()(wrappers.rhs().lhs(),free_fun());
    }
    else{
        lhs_ptr = &first_lhs.mat();
    }

    //Wrapping RHS
    utils::execute<ProdRhs>()(wrappers.rhs().rhs(),alloc_fun());
    typedef utils::transform<ProdRhs,wrapper_to_matrix,wrapper_to_matrix_fun> TmpTransformRhsT;
    typename TmpTransformRhsT::result_type tmprhs(TmpTransformRhsT::result(wrappers.rhs().rhs(),wrapper_to_matrix_fun()));
    typedef typename utils::make_shallow_copy<typename TmpTransformRhsT::result_type>::result_type MatType2;
    MatType2 real_rhs(utils::make_shallow_copy<typename TmpTransformRhsT::result_type>::result(tmprhs));
    typedef typename get_first_leave<MatType2>::result_type FirstRhsT;
    FirstRhsT first_rhs = get_first_leave<MatType2>::result(real_rhs);
    typename FirstRhsT::gpu_type * rhs_ptr = NULL;
    bool make_rhs_tmp = utils::n_handles(real_rhs)>1;

    if(make_rhs_tmp){
//        std::cout << "Creating RHS Temporary .." << std::endl;
        rhs_ptr = new typename FirstRhsT::gpu_type(first_rhs.mat().size1(), first_rhs.mat().size2());

        viennacl::generator::custom_operation op;
        op.add(FirstRhsT(*rhs_ptr) = real_rhs);
        op.execute();
        viennacl::backend::finish();
        utils::execute<ProdRhs>()(wrappers.rhs().rhs(),free_fun());
    }
    else{
        rhs_ptr = &first_rhs.mat();
    }

//    std::cout << "Execution of the Product" << std::endl;
    //Allocates result
    utils::execute<ResT>()(wrappers.lhs(),alloc_fun());
    generator::dummy_matrix<typename ResT::gpu_t> result(*wrappers.lhs().gpu_structure_ptr());
    viennacl::generator::custom_operation op(result = generator::prod(FirstLhsT(*lhs_ptr),FirstRhsT(*rhs_ptr)));
    op.execute();
    viennacl::backend::finish();


    if(make_lhs_tmp) delete lhs_ptr;
    else utils::execute<ProdLhs>()(wrappers.rhs().lhs(),free_fun());
    if(make_rhs_tmp) delete rhs_ptr;
    else utils::execute<ProdRhs>()(wrappers.rhs().rhs(),free_fun());

    //Transfer back result
    wrappers.lhs().transfer_back();
    wrappers.lhs().free();
}





/** @brief A dense matrix class - Multiple devices
*
* @tparam SCALARTYPE   The underlying scalar type (either float or double)
* @tparam F            Storage layout: Either row_major or column_major (at present only row_major is supported)
* @tparam ALIGNMENT   The internal memory size is given by (size()/ALIGNMENT + 1) * ALIGNMENT. ALIGNMENT must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
*/
template <class SCALARTYPE, typename F, unsigned int ALIGNMENT>
class multi_matrix
{

public:
  typedef multi_matrix<SCALARTYPE,F,ALIGNMENT> self_type;
  typedef matrix<SCALARTYPE, F, ALIGNMENT> gpu_matrix_type;
  typedef typename viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type>::cpu_t cpu_matrix_type;
  typedef typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<SCALARTYPE>::ResultType   value_type;
  typedef vcl_size_t                                                   size_type;

private:
  class block_t{
      friend class multi_matrix<SCALARTYPE,F,ALIGNMENT>;
  public:
      block_t(size_type row_offset
              , size_type col_offset
              , size_type size1
              , size_type size2) : row_offset_(row_offset), col_offset_(col_offset), matrix_(size1, size2){ }

      SCALARTYPE & operator()(size_type row_index, size_type col_index){
          return matrix_(row_index - row_offset_, col_index - col_offset_);
      }

      SCALARTYPE operator()(size_type row_index, size_type col_index) const {
          return matrix_(row_index - row_offset_, col_index - col_offset_);
      }

  private:
      size_type row_offset_;
      size_type col_offset_;
      cpu_matrix_type matrix_;
  };

private:
  typedef std::vector< std::vector<block_t> >  blocks_t;

private:

  size_t num_blocks_rows() const{
      return 1 + (rows_-1)/block_size_;
  }

  size_t num_blocks_columns() const{
      return  1 + (columns_-1)/block_size_;
  }

//  static size_t min_allocable_memory(viennacl::ocl::device const & d1, viennacl::ocl::device const & d2){
//      return std::min(d1.max_allocable_memory(),d2.max_allocable_memory());
//  }

  void init_blocks(){
      size_type num_blocks_row = num_blocks_rows();
      size_type num_blocks_col = num_blocks_columns();
      blocks_.resize(num_blocks_row);
      for(typename blocks_t::iterator it = blocks_.begin() ; it != blocks_.end() ; ++it){
          size_type row = it - blocks_.begin();
          size_type row_block_size = std::min(rows_ - row*block_size_, block_size_);
          it->reserve(num_blocks_col);
          for(size_type col = 0 ; col < num_blocks_col ; ++col){
              size_type col_block_size = std::min(columns_ - col*block_size_, block_size_);
//              std::cout << "Block size  " << row_block_size << "*" << col_block_size << std::endl;
              it->push_back(block_t(row*block_size_, col*block_size_, row_block_size,col_block_size));
          }
      }

  }

  size_t global_max_allocable_memory_size(){
      std::vector<size_t> max_allocable_sizes;
      std::vector<cl_device_id> devices = scheduler::devices();
      for(std::vector<cl_device_id>::const_iterator it = devices.begin(); it!=devices.end(); ++it){
              max_allocable_sizes.push_back(std::min(ocl::info<CL_DEVICE_GLOBAL_MEM_SIZE>(*it)/5,
                                                     ocl::info<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(*it)));
      }
      return  *std::min_element(max_allocable_sizes.begin(),max_allocable_sizes.end());
  }

public:
  /** @brief Creates the matrix with the given dimensions
  *
  * @param rows     Number of rows
  * @param columns  Number of columns
  */
  explicit multi_matrix(size_type rows, size_type columns) :
      rows_(rows), columns_(columns)
  {
    size_t max_allocable_size = global_max_allocable_memory_size();
    size_type max_block_size = viennacl::tools::roundDownToPreviousMultiple<vcl_size_t>(sqrt(max_allocable_size/(sizeof(SCALARTYPE))),256);
    block_size_ = std::min(max_block_size,rows/scheduler::n_devices());
    init_blocks();
  }

  /** @brief Returns the number of rows */
  const size_type & size1() const { return rows_;}
  /** @brief Returns the number of columns */
  const size_type & size2() const { return columns_; }

  //read-write access to an element of the matrix
  /** @brief Read-write access to a single element of the matrix
  */
  SCALARTYPE & operator()(size_type row_index, size_type col_index)
  {
      return blocks_.at(row_index/block_size_).at(col_index/block_size_)(row_index,col_index);
  }

  /** @brief Read access to a single element of the matrix
  */
  SCALARTYPE operator()(size_type row_index, size_type col_index) const
  {
      return blocks_.at(row_index/block_size_).at(col_index/block_size_)(row_index,col_index);
  }

  cpu_matrix_type const & block_matrix(size_type i, size_type j) const{
      return blocks_[i][j].matrix_;
  }


  //this = A * B and related (with trans())
  template <typename MatrixType1, typename MatrixType2>
  multi_matrix<SCALARTYPE, F, ALIGNMENT> & operator = (const matrix_expression< MatrixType1,
                                                                          MatrixType2,
                                                                          op_prod > & proxy)
  {
      typedef const typename MatrixType1::gpu_matrix_type gpu_matrix_type1;
      typedef const typename MatrixType2::gpu_matrix_type gpu_matrix_type2;
      typedef std::function<void (gpu_matrix_type & A, gpu_matrix_type1 const & B, gpu_matrix_type2 const & C)> fun_t;
      for(unsigned int row = 0 ; row < num_blocks_rows() ; ++row){
          for(unsigned int col = 0 ; col < num_blocks_columns() ; ++ col){
              //First product is not inplace
              viennacl::distributed::task * t1 = scheduler::create_task<gpu_matrix_type,gpu_matrix_type1,gpu_matrix_type2>(fun_t([](gpu_matrix_type & A, gpu_matrix_type1 const & B, gpu_matrix_type2 const & C) { A = viennacl::linalg::prod(B,C); }),
                                                                        viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_),
                                                                         viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type1>(proxy.lhs().block_matrix(row,0)),
                                                                         viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type2>(proxy.rhs().block_matrix(0,col)));
              t1->info("Matrix Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : Initial assignment " );

              //Inplace add of products
              for(unsigned int update = 1 ; update < proxy.lhs().num_blocks_columns() ; ++update){
                  viennacl::distributed::task * t2 = scheduler::create_task<gpu_matrix_type,gpu_matrix_type1,gpu_matrix_type2>(fun_t([](gpu_matrix_type & A, gpu_matrix_type1 const & B, gpu_matrix_type2 const & C) { A += viennacl::linalg::prod(B,C); }),
                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_),
                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type1>(proxy.lhs().block_matrix(row,update)),
                                                                            viennacl::distributed::utils::gpu_wrapper<gpu_matrix_type2>(proxy.rhs().block_matrix(update,col))
                                                                            );
                  t2->info("Matrix Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : " + viennacl::tools::to_string(update));
                  scheduler::connect(t1,t2);
                  t1 = t2;
              }

          }
      }
      scheduler::init();
    return *this;
  }


template<class MatrixType2>
generator::binary_matrix_expression_wrapper<self_type,generator::add_type,MatrixType2> operator+(MatrixType2 const & other){
    return generator::binary_matrix_expression_wrapper<self_type,generator::add_type,MatrixType2>(*this,other);
}

template<class MatrixType2>
generator::binary_matrix_expression_wrapper<self_type,generator::sub_type,MatrixType2> operator-(MatrixType2 const & other){
    return generator::binary_matrix_expression_wrapper<self_type,generator::sub_type,MatrixType2>(*this,other);
}


template <typename MatrixType1, typename MatrixType2, class OP>
multi_matrix<SCALARTYPE, F, ALIGNMENT> & operator = (const generator::binary_matrix_expression_wrapper< MatrixType1,
                                                                        OP,
                                                                        MatrixType2> & proxy)
{
    typedef generator::binary_matrix_expression_wrapper< MatrixType1, OP, MatrixType2> proxy_t;
    typedef generator::binary_matrix_expression_wrapper<self_type,generator::assign_type,proxy_t> operation_t;
    typedef typename utils::replace_type<operation_t,multi_to_wrapper>::result_type arg_t;
    typedef typename utils::replace_type<arg_t,wrapper_to_matrix>::result_type fun_t;
    typedef utils::transform<proxy_t,multi_to_wrapper,multi_to_wrapper_fun> transform_fun;
    typedef std::function<void (arg_t const &)> function_t;

    for(unsigned int row = 0 ; row < num_blocks_rows() ; ++row){
        for(unsigned int col = 0 ; col < num_blocks_columns() ; ++ col){
            arg_t arg(utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_),transform_fun::result(proxy,multi_to_wrapper_fun(row,col)));
            viennacl::distributed::task * t1 = scheduler::create_task(function_t(enqueue_op<arg_t>),arg);
            t1->info("Matrix " + proxy.op().repr() + " : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) );
        }
    }
    scheduler::init();
    return *this;
}

template <typename MatrixType1, typename MatrixType2, class ReduceType>
multi_matrix<SCALARTYPE, F, ALIGNMENT> & operator = (const generator::binary_matrix_expression_wrapper< MatrixType1,
                                                                        generator::reduce_type<ReduceType>,
                                                                        MatrixType2> & proxy)
{
    typedef generator::binary_matrix_expression_wrapper< MatrixType1, generator::prod_type<ReduceType>,MatrixType2> proxy_t;
    typedef generator::binary_matrix_expression_wrapper<self_type,generator::assign_type,proxy_t> operation_t;
    typedef typename utils::replace_type<operation_t,multi_to_wrapper>::result_type arg_t;
    typedef typename utils::replace_type<arg_t,wrapper_to_matrix>::result_type fun_t;

    typedef generator::binary_matrix_expression_wrapper<self_type,generator::inplace_add_type,proxy_t> operation_t2;
    typedef typename utils::replace_type<operation_t2,multi_to_wrapper>::result_type arg_t2;
    typedef typename utils::replace_type<arg_t,wrapper_to_matrix>::result_type fun_t2;

    typedef utils::transform<MatrixType1,multi_to_wrapper,multi_to_wrapper_fun> transform_fun_lhs;
    typedef utils::transform<MatrixType2,multi_to_wrapper,multi_to_wrapper_fun> transform_fun_rhs;

    typedef generator::binary_matrix_expression_wrapper< typename transform_fun_lhs::result_type,
                                                    generator::prod_type<ReduceType>,
                                                    typename transform_fun_rhs::result_type,
                                                    true> prod_type;


    typedef std::function<void (arg_t const &)> function_t;
    typedef std::function<void (arg_t  const &)> function_t2;

    for(unsigned int row = 0 ; row < num_blocks_rows() ; ++row){
        for(unsigned int col = 0 ; col < num_blocks_columns() ; ++ col){
            //First product is not inplace
            prod_type prod(transform_fun_lhs::result(proxy.lhs(),multi_to_wrapper_fun(row,0))
                           ,transform_fun_rhs::result(proxy.rhs(),multi_to_wrapper_fun(0,col)));
            arg_t arg(utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_),prod);
            viennacl::distributed::task * t1 = scheduler::create_task(function_t(enqueue_op_prod<arg_t>),arg);
            t1->info("Matrix Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : Initial assignment " );
//            Inplace add of products
            for(unsigned int update = 1 ; update < get_first_leave<MatrixType1>::result(proxy.lhs()).num_blocks_columns() ; ++update){
                prod_type prod2(transform_fun_lhs::result(proxy.lhs(),multi_to_wrapper_fun(row,update)),transform_fun_rhs::result(proxy.rhs(),multi_to_wrapper_fun(update,col)));
                arg_t arg2(utils::gpu_wrapper<gpu_matrix_type>(blocks_[row][col].matrix_),prod2);
                viennacl::distributed::task * t2 = scheduler::create_task(function_t2(enqueue_op_prod<arg_t>),arg2);
                t2->info("Matrix Product : Block " + viennacl::tools::to_string(row) +  "," + viennacl::tools::to_string(col) + " : " + viennacl::tools::to_string(update));
                scheduler::connect(t1,t2);
                t1 = t2;
            }

        }
    }
    scheduler::init();
    return *this;
}

  template <typename SCALARTYPE1, typename F1, typename SCALARTYPE2, typename F2, unsigned int ALIGNMENT2>
  friend void viennacl::copy(const distributed::cpu_matrix<SCALARTYPE1, F1> & mat,
            multi_matrix<SCALARTYPE2, F2, ALIGNMENT2> & gpu_matrix );

private:
  size_type rows_;
  size_type columns_;
  size_type block_size_;
  blocks_t blocks_;
}; //multi_matrix

template<class SCALARTYPE, typename F, unsigned int ALIGNMENT>
std::ostream & operator<<(std::ostream & s, const multi_matrix<SCALARTYPE, F, ALIGNMENT> & gpu_matrix)
{
    typedef typename matrix<SCALARTYPE, F, ALIGNMENT>::size_type      size_type;
    s << "[" << gpu_matrix.size1() << "," << gpu_matrix.size2() << "]";
    s << "(";
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      s << "(";
      for (size_type j = 0; j < gpu_matrix.size2(); ++j)
      {
        s << gpu_matrix(i, j);
        if (j < gpu_matrix.size2() - 1)
          s << ",";
      }
      s << ")";
      if (i < gpu_matrix.size1() - 1)
        s << ",";
    }
    s << ")";
    return s;
}


} //namespace distributed

} //namespace viennacl

#endif
