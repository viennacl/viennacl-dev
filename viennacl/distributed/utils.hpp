#ifndef VIENNACL_DISTRIBUTED_UTILS_HPP_
#define VIENNACL_DISTRIBUTED_UTILS_HPP_

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

/** @file utils.hpp
    @brief Implementation of several utils
*/

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/generator/forwards.h"

#include "viennacl/distributed/cpu_matrix.hpp"

#include <set>

namespace viennacl{

namespace distributed{

namespace utils{

template <class INT_TYPE>
INT_TYPE roundDownToPreviousMultiple(INT_TYPE to_reach, INT_TYPE base)
{
  if (to_reach % base == 0) return to_reach;
  return (to_reach / base) * base;
}

template<class ScalarType>
vcl_size_t matrix_block_size(){

    vcl_size_t mem_chunk_size = 128*1024*1024;
    return roundDownToPreviousMultiple<vcl_size_t>(sqrt(mem_chunk_size/(sizeof(ScalarType))),256);
}

template<class ScalarType>
vcl_size_t vector_block_size(){
    return matrix_block_size<ScalarType>()*matrix_block_size<ScalarType>();
}

template<class T>
struct get_cpu_type;

template<class SCALARTYPE, class F, unsigned int Alignment>
struct get_cpu_type<viennacl::matrix<SCALARTYPE,F,Alignment> >{
    typedef cpu_matrix<SCALARTYPE, F> type;
};

template<class SCALARTYPE, class F, unsigned int Alignment>
struct get_cpu_type<const viennacl::matrix<SCALARTYPE,F,Alignment> >{
    typedef const cpu_matrix<SCALARTYPE, F> type;
};


template<class SCALARTYPE, unsigned int Alignment>
struct get_cpu_type<viennacl::vector<SCALARTYPE,Alignment> >{
    typedef std::vector<SCALARTYPE> type;
};

template<class SCALARTYPE,unsigned int Alignment>
struct get_cpu_type<const viennacl::vector<SCALARTYPE,Alignment> >{
    typedef const std::vector<SCALARTYPE> type;
};

/** @brief Storage for lazy gpu allocation */

template<class T, class CpuT>
struct alloc_impl;

template<class ScalarType, class F, class CpuT>
struct alloc_impl<viennacl::matrix<ScalarType, F>, CpuT>{
    viennacl::matrix<ScalarType, F>* operator()(CpuT const & cpu_data){
        size_t size1=cpu_data.size1(), size2=cpu_data.size2();
        viennacl::matrix<ScalarType, F>* p = new viennacl::matrix<ScalarType, F>(size1, size2);
        cl_mem h = p->handle().opencl_handle();
        clEnqueueWriteBuffer(viennacl::ocl::current_context().get_queue().handle().get(),h,true,0,size1*size2,&cpu_data(0,0),0,NULL,NULL);
        return p;
    }
};

template<class ScalarType, class F, class CpuT>
struct alloc_impl<const viennacl::matrix<ScalarType, F>, CpuT>{
    const viennacl::matrix<ScalarType, F>* operator()(CpuT const & cpu_data){
        size_t size1=cpu_data.size1(), size2=cpu_data.size2();
        const viennacl::matrix<ScalarType, F>* p = new viennacl::matrix<ScalarType, F>(size1, size2);
        cl_mem h = p->handle().opencl_handle();
        clEnqueueWriteBuffer(viennacl::ocl::current_context().get_queue().handle().get(),h,true,0,size1*size2,&cpu_data(0,0),0,NULL,NULL);
        return p;
    }
};

//template<class ScalarType, class F, class CpuT>
//struct alloc_impl<const viennacl::matrix<ScalarType, F>, CpuT>{
//    viennacl::matrix<ScalarType, F>* operator()(viennacl::ocl::context &ctxt, CpuT const & cpu_data){
//        return new viennacl::matrix<ScalarType, F>(cpu_data.size1(), cpu_data.size2(), &cpu_data(0,0), ctxt);
//    }
//};


//template<class ScalarType, class F, class CpuT>
//viennacl::matrix<ScalarType, F>* alloc_impl<viennacl::matrix<ScalarType, F>, CpuT>(viennacl::ocl::context & ctxt, CpuT const & cpu_data){
//    return new viennacl::matrix<ScalarType, F>(cpu_data.size1(), cpu_data.size2(), &cpu_data(0,0), ctxt);
//}

//template<class ScalarType>
//viennacl::vector<ScalarType>* alloc_impl(viennacl::ocl::context & ctxt){
//    return new viennacl::vector<ScalarType>(cpu_data.size1(), cpu_data.size2(), &cpu_data(0,0), ctxt);
//}

template<class T>
class gpu_wrapper{
public:
    typedef typename get_cpu_type<T>::type cpu_t;
    typedef T gpu_t;

    gpu_wrapper(cpu_t const & _cpu_data) : cpu_data(_cpu_data){
    }

    gpu_wrapper(gpu_wrapper const & other) : cpu_data(other.cpu_data), gpu_structure_(other.gpu_structure_){
        assert(gpu_structure_.get() == NULL);
    }

    void alloc() const {
        gpu_structure_.reset(alloc_impl<T,cpu_t>()(cpu_data));
    }

    void free() const{
        gpu_structure_.reset();
    }

    void transfer_back() const {
        size_t internal_size = cpu_data.size1() * cpu_data.size2();
        clEnqueueReadBuffer(viennacl::ocl::current_context().get_queue().handle().get(),gpu_structure_->handle().opencl_handle(),true,0,internal_size,(void*)&cpu_data(0,0),0,NULL,NULL);
    }

    T * gpu_structure_ptr() const {
        return gpu_structure_.get();
    }


private:
    cpu_t const & cpu_data;
    mutable viennacl::tools::shared_ptr<T> gpu_structure_;
};


template<class T, template<class> class Fun>
struct replace_type{
    typedef typename Fun<T>::type result_type;
};

template<class LHS, class OP, class RHS, bool deep_copy, template<class> class Fun>
struct replace_type<generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy>,Fun >{
private:
    typedef typename replace_type<LHS,Fun>::result_type lhs_result_type;
    typedef typename replace_type<RHS,Fun>::result_type rhs_result_type;
public:
    typedef generator::matrix_expression_wrapper<lhs_result_type, OP, rhs_result_type,true> result_type;
};

template<class T, template<class> class TypeFun, class Fun>
struct transform{
    typedef typename TypeFun<T>::type result_type;
    static result_type result(T const & t, Fun fun){
        return fun(t);
    }
};

template<class LHS, class OP, class RHS, bool deep_copy, template<class> class TypeFun, class Fun>
struct transform<generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy>, TypeFun, Fun>{
    typedef typename transform<LHS,TypeFun, Fun>::result_type lhs_result_type;
    typedef typename transform<RHS,TypeFun, Fun>::result_type rhs_result_type;
    typedef generator::matrix_expression_wrapper<lhs_result_type, OP, rhs_result_type, true> result_type;

    static result_type result(generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> const & mat, Fun fun){
        return result_type(transform<LHS,TypeFun,Fun>::result(mat.lhs(), fun),transform<RHS,TypeFun,Fun>::result(mat.rhs(),fun));
    }
};

template<class T>
struct make_deep_copy{
    typedef T result_type;
    static result_type result(T const & t){
        return t;
    }
};

template<class LHS, class OP, class RHS, bool deep_copy>
struct make_deep_copy<generator::matrix_expression_wrapper<LHS,OP,RHS,deep_copy> >{
    typedef typename make_deep_copy<LHS>::result_type lhs_result_type;
    typedef typename make_deep_copy<RHS>::result_type rhs_result_type;
    typedef generator::matrix_expression_wrapper<lhs_result_type, OP, rhs_result_type, true> result_type;

    static result_type result(generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> const & mat){
        return result_type(make_deep_copy<LHS>::result(mat.lhs()),make_deep_copy<RHS>::result(mat.rhs()));
    }
};


template<class T>
struct make_shallow_copy{
    typedef T result_type;
    static result_type const & result(T const & t){
        return t;
    }
};

template<class LHS, class OP, class RHS, bool deep_copy>
struct make_shallow_copy<generator::matrix_expression_wrapper<LHS,OP,RHS,deep_copy> >{
    typedef typename make_shallow_copy<LHS>::result_type lhs_result_type;
    typedef typename make_shallow_copy<RHS>::result_type rhs_result_type;
    typedef generator::matrix_expression_wrapper<lhs_result_type, OP, rhs_result_type, false> result_type;

    static result_type result(generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> const & mat){
        return result_type(make_shallow_copy<LHS>::result(mat.lhs()),make_shallow_copy<RHS>::result(mat.rhs()));
    }
};




template<class T>
struct execute{
    template<class Fun>
    void operator()(T const & t, Fun fun){
        return fun(t);
    }
};

template<class LHS, class OP, class RHS, bool deep_copy>
struct execute<generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> >{
    template<class Fun>
    void operator()(generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> const & mat, Fun fun){
        execute<LHS>()(mat.lhs(),fun);
        execute<RHS>()(mat.rhs(),fun);
    }
};

template<class T>
void fill_handle(T const & t, std::set<cl_mem> & mems){
    mems.insert(t.mat().handle().opencl_handle());
}

template<class LHS, class OP, class RHS, bool deep_copy>
void fill_handle(generator::matrix_expression_wrapper<LHS,OP,RHS, deep_copy> const & t, std::set<cl_mem> & mems){
    fill_handle(t.lhs(),mems);
    fill_handle(t.rhs(),mems);
}

template<class T>
unsigned int n_handles(T const & t){
    std::set<cl_mem> garbage;
    fill_handle(t,garbage);
    return garbage.size();
}

}

}

}

#endif // VIENNACL_DISTRIBUTED_UTILS_HPP_
