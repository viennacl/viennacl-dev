#ifndef VIENNACL_GENERATOR_OVERLOADS_HPP
#define VIENNACL_GENERATOR_OVERLOADS_HPP

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


/** @file viennacl/generator/overloads.hpp
    @brief Contains the overloads used by the user interface
*/


#include "viennacl/meta/enable_if.hpp"
#include "viennacl/generator/forwards.h"
//#include "viennacl/forwards.h"
#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/vector.hpp"
#include <set>

namespace viennacl{

  namespace generator{



    /** @brief Simple wrapper from viennacl::vector
     *
     *   Just holds a reference to a viennacl vector and provides the corresponding operator overloads.
     *  Meant to be merged with viennacl::vector once all the kernels are dynamically generated.
     */
    template<typename SCALARTYPE>
    class vector{
        typedef vector<SCALARTYPE> self_type;
        typedef viennacl::vector<SCALARTYPE> vcl_t;
      public:

        vector(vcl_t const & vec): vec_(vec){ }

        vcl_t const & get() const{ return vec_; }

        template<typename RHS_TYPE>
        binary_vector_expression<typename to_sym<self_type>::type, assign_type, typename to_sym<RHS_TYPE>::type >
        operator= ( RHS_TYPE const & rhs ){
          return binary_vector_expression<typename to_sym<self_type>::type,assign_type,typename to_sym<RHS_TYPE>::type >(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_vector_expression<typename to_sym<self_type>::type, inplace_scal_mul_type, typename to_sym<RHS_TYPE>::type >
        operator*= ( RHS_TYPE const & rhs ){
          return binary_vector_expression<typename to_sym<self_type>::type,inplace_scal_mul_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_vector_expression<typename to_sym<self_type>::type, inplace_scal_div_type, typename to_sym<RHS_TYPE>::type >
        operator/= ( RHS_TYPE const & rhs ){
          return binary_vector_expression<typename to_sym<self_type>::type,inplace_scal_div_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_vector_expression<typename to_sym<self_type>::type, inplace_add_type, typename to_sym<RHS_TYPE>::type >
        operator+= ( RHS_TYPE const & rhs ){
          return binary_vector_expression<typename to_sym<self_type>::type,inplace_add_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_vector_expression<typename to_sym<self_type>::type, inplace_sub_type, typename to_sym<RHS_TYPE>::type >
        operator-= ( RHS_TYPE const & rhs ){
          return binary_vector_expression<typename to_sym<self_type>::type,inplace_sub_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }
      private:
        vcl_t const & vec_;
    };

    /** @brief Constant vector wrapper
     *
     *  Does not allocate any memory
     *
     *  @todo make it take any std::string instead of a template integer...
     */
    template<unsigned int N>
    class constant_vector{
      public:
        static const unsigned int val = N;
    };

    /** @brief Wrap an interface vector to an internal symbolic vector */
    template<class ScalarType>
    struct to_sym<vector<ScalarType> >{
        typedef symbolic_vector<ScalarType, vector_handle_accessor, index_set> type;
        static type result(vector<ScalarType> const & t) { return type(t.get().size(),vector_handle_accessor(t.get().handle()), index_set()); }
    };

    template<unsigned int N>
    struct to_sym<constant_vector<N> >{
        typedef symbolic_constant type;
        static type result(constant_vector<N> const &) { return type(utils::to_string(N)); }
    };


    /** @brief Simple wrapper from viennacl::scalar
     *
     *   Just holds a reference to a viennacl scalar and provides the corresponding operator overloads.
     *  Meant to be merged with viennacl::vector once all the kernels are dynamically generated.
     */
    template<class ScalarType>
    class scalar{
        typedef scalar<ScalarType> self_type;
      public:
        typedef viennacl::scalar<ScalarType> vcl_t;

        scalar(vcl_t const & scal): scal_(scal){ }

        vcl_t const & get() const{ return scal_; }

        template<typename RHS_TYPE>
        binary_scalar_expression<typename to_sym<self_type>::type, assign_type, typename to_sym<RHS_TYPE>::type  >
        operator= ( RHS_TYPE const & rhs ){
          return binary_scalar_expression<typename to_sym<self_type>::type,assign_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_scalar_expression<typename to_sym<self_type>::type, inplace_scal_mul_type, typename to_sym<RHS_TYPE>::type  >
        operator*= ( RHS_TYPE const & rhs ){
          return binary_scalar_expression<typename to_sym<self_type>::type,inplace_scal_mul_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_scalar_expression<typename to_sym<self_type>::type, inplace_scal_div_type, typename to_sym<RHS_TYPE>::type  >
        operator/= ( RHS_TYPE const & rhs ){
          return binary_scalar_expression<typename to_sym<self_type>::type,inplace_scal_div_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_scalar_expression<typename to_sym<self_type>::type, inplace_add_type, typename to_sym<RHS_TYPE>::type  >
        operator+= ( RHS_TYPE const & rhs ){
          return binary_scalar_expression<typename to_sym<self_type>::type,inplace_add_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }

        template<typename RHS_TYPE>
        binary_scalar_expression<typename to_sym<self_type>::type, inplace_sub_type, typename to_sym<RHS_TYPE>::type  >
        operator-= ( RHS_TYPE const & rhs ){
          return binary_scalar_expression<typename to_sym<self_type>::type,inplace_sub_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),make_sym(rhs));
        }
      private:
        vcl_t const & scal_;
    };


    /** @brief Wrap an interface scalar to an internal symbolic scalar */
    template<class ScalarType>
    struct to_sym<scalar<ScalarType> >{
        typedef gpu_symbolic_scalar<ScalarType> type;
        static type result(scalar<ScalarType> const & t) { return t.get(); }
    };

    template<typename ScalarType>
    struct to_sym<ScalarType, typename viennacl::enable_if<is_primitive_type<ScalarType>::value >::type>{
        typedef cpu_symbolic_scalar<ScalarType> type;
        static type result(ScalarType const & t){ return t; }
    };


    /** @brief Simple wrapper from viennacl::matrix
     *
     *   Just holds a reference to a viennacl vector and provides the corresponding operator overloads.
     *  Meant to be merged with viennacl::vector once all the kernels are dynamically generated.
     */
    template<class VCL_MATRIX>
    class matrix{
        typedef matrix<VCL_MATRIX> self_type;
      public:

        typedef VCL_MATRIX vcl_t;

        matrix(VCL_MATRIX & mat) : mat_(mat){ }

        vcl_t const & get() const{
          return mat_;
        }

        template<typename RHS_TYPE>
        binary_matrix_expression<typename to_sym<self_type>::type, assign_type, typename to_sym<RHS_TYPE>::type  >
        operator= ( RHS_TYPE const & rhs ){
          return binary_matrix_expression<typename to_sym<self_type>::type,assign_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),rhs);
        }

        template<typename RHS_TYPE>
        binary_matrix_expression<typename to_sym<self_type>::type, inplace_scal_mul_type, typename to_sym<RHS_TYPE>::type  >
        operator*= ( RHS_TYPE const & rhs ){
          return binary_matrix_expression<typename to_sym<self_type>::type,inplace_scal_mul_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),rhs);
        }

        template<typename RHS_TYPE>
        binary_matrix_expression<typename to_sym<self_type>::type, inplace_scal_div_type, typename to_sym<RHS_TYPE>::type  >
        operator/= ( RHS_TYPE const & rhs ){
          return binary_matrix_expression<typename to_sym<self_type>::type,inplace_scal_div_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),rhs);
        }

        template<typename RHS_TYPE>
        binary_matrix_expression<typename to_sym<self_type>::type, inplace_add_type, typename to_sym<RHS_TYPE>::type  >
        operator+= ( RHS_TYPE const & rhs ){
          return binary_matrix_expression<typename to_sym<self_type>::type,inplace_add_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),rhs);
        }

        template<typename RHS_TYPE>
        binary_matrix_expression<typename to_sym<self_type>::type, inplace_sub_type, typename to_sym<RHS_TYPE>::type  >
        operator-= ( RHS_TYPE const & rhs ){
          return binary_matrix_expression<typename to_sym<self_type>::type,inplace_sub_type,typename to_sym<RHS_TYPE>::type>(make_sym(*this),rhs);
        }
      private:
        VCL_MATRIX const & mat_;
    };


    /** @brief Wrap an interface matrix to an internal symbolic matrix */
    template<class VCL_MATRIX_T>
    struct to_sym<matrix<VCL_MATRIX_T> >{
        typedef symbolic_matrix<VCL_MATRIX_T, matrix_handle_accessor, index_set, index_set> type;
        static type result(matrix<VCL_MATRIX_T> const & t) { return type(t.get().internal_size1(), t.get().internal_size2(), matrix_handle_accessor(t.get().handle()), index_set(), index_set()); }
    };


    /** @brief Some traits for vector expressions that have a symbolic equivalent */
    template<class T>
    struct is_vector_expression_t{ enum { value = 0 }; };
    template<typename ScalarType>
    struct is_vector_expression_t<vector<ScalarType> >{ enum { value = 1}; };
    template<typename ScalarType, class Elements, class Index>
    struct is_vector_expression_t<symbolic_vector<ScalarType, Elements, Index> >{ enum { value = 1}; };
    template<>
    struct is_vector_expression_t<index_set>{ enum { value = 1}; };
    template<unsigned int N>
    struct is_vector_expression_t<constant_vector<N> >{ enum { value = 1}; };
    template<class LHS, class OP, class RHS>
    struct is_vector_expression_t<binary_vector_expression<LHS,OP,RHS> >{ enum { value = 1}; };
    template<class SUB, class OP>
    struct is_vector_expression_t<unary_vector_expression<SUB,OP> >{ enum { value = 1}; };

    /** @brief Some traits for scalar expressions that have a symbolic equivalent */
    template<class T>
    struct is_scalar_expression_t{ enum { value = is_primitive_type<T>::value }; };
    template<class ScalarType>
    struct is_scalar_expression_t<scalar<ScalarType> >{ enum { value = 1}; };
    template<>
    struct is_scalar_expression_t<float> { enum { value = 1 }; };
    template<class LHS, class OP, class RHS>
    struct is_scalar_expression_t<binary_scalar_expression<LHS,OP,RHS> >{ enum { value = 1}; };
    template<class SUB, class OP>
    struct is_scalar_expression_t<unary_scalar_expression<SUB,OP> >{ enum { value = 1}; };

    /** @brief Some traits for matrix expressions that have a symbolic equivalent */
    template<class T>
    struct is_matrix_expression_t{ enum { value = 0}; };
    template<class VCL_MATRIX>
    struct is_matrix_expression_t<matrix<VCL_MATRIX> >{ enum { value = 1}; };
    template<class VCL_MATRIX, class ELEMENTS, class ACCESS_ROW, class ACCESS_COL>
    struct is_matrix_expression_t<symbolic_matrix<VCL_MATRIX, ELEMENTS, ACCESS_ROW, ACCESS_COL> >{ enum { value = 1}; };
    //template<class Scalartype, class F>
    //struct is_matrix_expression_t<viennacl::distributed::multi_matrix<Scalartype, F> >{ enum { value = 1}; };
    template<class LHS, class OP, class RHS>
    struct is_matrix_expression_t<binary_matrix_expression<LHS,OP,RHS> >{ enum { value = 1}; };
    //template<class T>
    //struct is_matrix_expression_t<viennacl::distributed::utils::gpu_wrapper<T> > { enum { value = 1 }; };
    template<class SUB, class OP>
    struct is_matrix_expression_t<unary_matrix_expression<SUB,OP> >{ enum { value = 1}; };


    /** @brief generic functor to infer type for common operators such as +, -
     *
     *  Leads to the creation of only one implementation of the corresponding operators.
     */
    template<class LHS, class OP, class RHS, bool create_vector, bool create_scalar, bool create_matrix>
    struct convert_to_binary_expr;
    template<class LHS, class OP, class RHS>
    struct convert_to_binary_expr<LHS,OP,RHS,true,false,false>{ typedef binary_vector_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };
    template<class LHS, class OP, class RHS>
    struct convert_to_binary_expr<LHS,OP,RHS,false,true,false>{ typedef binary_scalar_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };
    template<class LHS, class OP, class RHS>
    struct convert_to_binary_expr<LHS,OP,RHS,false,false,true>{ typedef binary_matrix_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };



    /** @brief Some traits for operators that have a symbolic equivalent */
    template<class T>
    struct is_operator{ enum{ value = 0}; };
    template<> struct is_operator<assign_type>{ enum { value = 1}; };
    template<> struct is_operator<add_type>{ enum { value = 1}; };
    template<> struct is_operator<inplace_add_type>{ enum { value = 1}; };
    template<> struct is_operator<sub_type>{ enum { value = 1}; };
    template<> struct is_operator<inplace_sub_type>{ enum { value = 1}; };
    template<> struct is_operator<mul_type>{ enum { value = 1}; };
    template<> struct is_operator<inplace_scal_mul_type>{ enum { value = 1}; };
    template<> struct is_operator<div_type>{ enum { value = 1}; };
    template<> struct is_operator<inplace_scal_div_type>{ enum { value = 1}; };

    template<class T>
    struct is_leaf{ enum{ value = 0}; };
    template<class ScalarType> struct is_leaf<vector<ScalarType> >{ enum { value = 1 }; };
    template<class ScalarType> struct is_leaf<scalar<ScalarType> >{ enum { value = 1 }; };
    template<class VCL_MATRIX> struct is_leaf<matrix<VCL_MATRIX> >{ enum { value = 1 }; };


    template<class LHS, class RHS> struct create_vector{
        enum{  value= (is_vector_expression_t<LHS>::value && is_scalar_expression_t<RHS>::value)
               || (is_scalar_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
               || (is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value) };
    };

    template<class LHS, class RHS> struct create_scalar{
        enum{  value= (is_scalar_expression_t<LHS>::value && is_scalar_expression_t<RHS>::value) };
    };


    template<class LHS, class RHS> struct create_matrix{
        enum{  value= (is_matrix_expression_t<LHS>::value && is_scalar_expression_t<RHS>::value)
               || (is_scalar_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
               || (is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value) };
    };




    /////////////////////////////////////////
    ///// BINARY OPERATORS
    ////////////////////////////////////////



    /** @brief elementwise product */
    template<class LHS, class RHS>
    typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
    ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
    ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
    ,typename convert_to_binary_expr<LHS,mul_type,RHS
    ,create_vector<LHS,RHS>::value
    ,create_scalar<LHS,RHS>::value
    ,create_matrix<LHS,RHS>::value>::type>::type
    element_prod(LHS const & lhs, RHS const & rhs){
      return typename convert_to_binary_expr<LHS,mul_type,RHS
          ,create_vector<LHS,RHS>::value
          ,create_scalar<LHS,RHS>::value
          ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
    }

    /** @brief elementwise division */
    template<class LHS, class RHS>
    typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
    ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
    ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
    ,typename convert_to_binary_expr<LHS,div_type,RHS
    ,create_vector<LHS,RHS>::value
    ,create_scalar<LHS,RHS>::value
    ,create_matrix<LHS,RHS>::value>::type>::type
    element_div(LHS const & lhs, RHS const & rhs){
      return typename convert_to_binary_expr<LHS,div_type,RHS
          ,create_vector<LHS,RHS>::value
          ,create_scalar<LHS,RHS>::value
          ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
    }


/** @brief macro for elementwise operators */
#define CREATE_ELEMENTWISE_OPERATOR(function_name, symbolic_type) \
  template<class LHS, class RHS>\
  typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)\
  ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)\
  ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)\
  ,typename convert_to_binary_expr<LHS,symbolic_type,RHS\
  ,create_vector<LHS,RHS>::value\
  ,create_scalar<LHS,RHS>::value\
  ,create_matrix<LHS,RHS>::value>::type>::type function_name(LHS const & lhs, RHS const & rhs){\
  return typename convert_to_binary_expr<LHS,symbolic_type,RHS\
  ,create_vector<LHS,RHS>::value\
  ,create_scalar<LHS,RHS>::value\
  ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));\
  }\

    CREATE_ELEMENTWISE_OPERATOR(operator+,add_type)
    CREATE_ELEMENTWISE_OPERATOR(operator-,sub_type)
    CREATE_ELEMENTWISE_OPERATOR(operator>,sup_type)
    CREATE_ELEMENTWISE_OPERATOR(operator>=,supeq_type)
    CREATE_ELEMENTWISE_OPERATOR(operator<,inf_type)
    CREATE_ELEMENTWISE_OPERATOR(operator<=,infeq_type)
    //CREATE_ELEMENTWISE_OPERATOR(operator==,eqto_type)
    CREATE_ELEMENTWISE_OPERATOR(operator!=,neqto_type)
#undef CREATE_ELEMENTWISE_OPERATOR

    //Prod needs a special treatment to avoid ambiguities
    template<class LHS, class RHS>
    typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
    ,typename convert_to_binary_expr<LHS,mul_type,RHS
    ,create_vector<LHS,RHS>::value
    ,create_scalar<LHS,RHS>::value
    ,create_matrix<LHS,RHS>::value>::type>::type
    operator*(LHS const & lhs, RHS const & rhs){
      return typename convert_to_binary_expr<LHS,mul_type,RHS
          ,create_vector<LHS,RHS>::value
          ,create_scalar<LHS,RHS>::value
          ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
    }

    template<class LHS, class RHS>
    typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
    ,typename convert_to_binary_expr<LHS,div_type,RHS
    ,create_vector<LHS,RHS>::value
    ,create_scalar<LHS,RHS>::value
    ,create_matrix<LHS,RHS>::value>::type>::type
    operator/(LHS const & lhs, RHS const & rhs){
      return typename convert_to_binary_expr<LHS,div_type,RHS
          ,create_vector<LHS,RHS>::value
          ,create_scalar<LHS,RHS>::value
          ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
    }

    template<class LHS, class RHS>
    typename viennacl::enable_if<is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value
    ,binary_scalar_expression<typename to_sym<LHS>::type,reduce_type<add_type>, typename to_sym<RHS>::type> >::type
    inner_prod(LHS const & lhs, RHS const & rhs)
    {
      return binary_scalar_expression<typename to_sym<LHS>::type,reduce_type<add_type>, typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
    }


    template<class LHS, class RHS>
    typename viennacl::enable_if<is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value
    ,binary_matrix_expression<typename to_sym<LHS>::type,reduce_type<add_type>,typename to_sym<RHS>::type> >::type
    prod(LHS const & lhs, RHS const & rhs)
    {
      return binary_matrix_expression<typename to_sym<LHS>::type,reduce_type<add_type>,typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
    }


    template<class LHS, class RHS>
    typename viennacl::enable_if<is_matrix_expression_t<LHS>::value && is_vector_expression_t<RHS>::value
    ,binary_vector_expression<typename to_sym<LHS>::type,reduce_type<add_type>,typename to_sym<RHS>::type> >::type
    prod(LHS const & lhs, RHS const & rhs)
    {
      return binary_vector_expression<typename to_sym<LHS>::type,reduce_type<add_type>,typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
    }


    ///////////////////////////////////////////
    /////// UNARY OPERATORS
    //////////////////////////////////////////

    template<class SUB, class OP, bool create_vector, bool create_scalar, bool create_matrix>
    struct convert_to_unary_expr;
    template<class SUB, class OP>
    struct convert_to_unary_expr<SUB,OP,true,false,false>{ typedef unary_vector_expression<typename to_sym<SUB>::type, OP> type; };
    template<class SUB, class OP>
    struct convert_to_unary_expr<SUB,OP,false,true,false>{ typedef unary_scalar_expression<typename to_sym<SUB>::type, OP> type; };
    template<class SUB, class OP>
    struct convert_to_unary_expr<SUB,OP,false,false,true>{ typedef unary_matrix_expression<typename to_sym<SUB>::type, OP> type; };

    template<class T>
    typename viennacl::enable_if<is_matrix_expression_t<T>::value, unary_matrix_expression<typename to_sym<T>::type,trans_type> >::type
    trans(T const & t){
      return unary_matrix_expression<typename to_sym<T>::type,trans_type>(make_sym(t));
    }

    template<class ScalarType, class F, unsigned int A>
    symbolic_vector<ScalarType
                      ,vector_handle_accessor
                    ,binary_vector_expression<cpu_symbolic_scalar<long unsigned int>,mul_type,index_set> >
    diag(viennacl::generator::matrix<viennacl::matrix<ScalarType, F, A> > const & m){
      return symbolic_vector<ScalarType
          ,vector_handle_accessor
        ,binary_vector_expression<cpu_symbolic_scalar<long unsigned int>,mul_type,index_set> >(m.get().size1(), vector_handle_accessor(m.get().handle()), (m.get().size1()+1)*index_set());
    }


    template<class ScalarType>
    symbolic_matrix<viennacl::matrix<ScalarType,viennacl::row_major>, matrix_diag_accessor, index_set, index_set>
    diag(viennacl::generator::vector<ScalarType> const & v){
      return symbolic_matrix<viennacl::matrix<ScalarType,viennacl::row_major>, matrix_diag_accessor, index_set, index_set>(v.get().size(), v.get().size(), matrix_diag_accessor(v.get().handle()),index_set(), index_set());
    }

    template<class ScalarType, class F>
    symbolic_matrix<viennacl::matrix<ScalarType,F>,  matrix_repmat_accessor<viennacl::matrix<ScalarType,F> >, index_set, index_set>
    repmat(viennacl::generator::matrix<viennacl::matrix<ScalarType,F> > const & m, unsigned int repsize1, unsigned int repsize2){
      size_t size1 = m.get().internal_size1();
      size_t size2 = m.get().internal_size2();
      return symbolic_matrix<viennacl::matrix<ScalarType,F>,  matrix_repmat_accessor<viennacl::matrix<ScalarType,F> >, index_set, index_set>(size1*repsize1, size2*repsize2, matrix_repmat_accessor<viennacl::matrix<ScalarType,F> >(size1,size2,m.get().handle()), index_set(), index_set());
    }

    template<class ScalarType>
    symbolic_matrix<viennacl::matrix<ScalarType,viennacl::row_major>,  matrix_repmat_accessor<viennacl::matrix<ScalarType,viennacl::row_major> >, index_set, index_set>
    repmat(viennacl::generator::vector<ScalarType> const & v, unsigned int repsize1, unsigned int repsize2){
      size_t size = v.get().size();
      return symbolic_matrix<viennacl::matrix<ScalarType,viennacl::row_major>,  matrix_repmat_accessor<viennacl::matrix<ScalarType,viennacl::row_major> >, index_set, index_set>(size*repsize1, repsize2, matrix_repmat_accessor<viennacl::matrix<ScalarType,viennacl::row_major> >(size,1,v.get().handle()), index_set(), index_set());
    }

    template<class T>
    typename viennacl::enable_if<is_scalar_expression_t<T>::value ||is_vector_expression_t<T>::value||is_matrix_expression_t<T>::value
    ,typename convert_to_unary_expr<T,unary_sub_type,is_vector_expression_t<T>::value
    ,is_scalar_expression_t<T>::value
    ,is_matrix_expression_t<T>::value>::type>::type
    operator-(T const & t)
    {
      return typename convert_to_unary_expr<T,unary_sub_type
          ,is_vector_expression_t<T>::value
          ,is_scalar_expression_t<T>::value
          ,is_matrix_expression_t<T>::value>::type(make_sym(t));
    }

    template<class OP_REDUCE, class T>
    typename viennacl::enable_if<is_vector_expression_t<T>::value
    ,binary_scalar_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant > >::type
    reduce(T const & t){
      return binary_scalar_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant >(make_sym(t), symbolic_constant("1"));
    }


    template<class OP_REDUCE, class T>
    typename viennacl::enable_if<is_matrix_expression_t<T>::value
    ,binary_vector_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant > >::type
    reduce_rows(T const & t){
      return binary_vector_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant >(make_sym(t), symbolic_constant("1"));
    }

    template<class OP_REDUCE, class T>
    typename viennacl::enable_if<is_matrix_expression_t<T>::value
    ,binary_vector_expression<
    unary_matrix_expression<typename to_sym<T>::type,trans_type>
    ,reduce_type<OP_REDUCE>
    ,symbolic_constant > >::type
    reduce_cols(T const & t){
      return binary_vector_expression<
          unary_matrix_expression<typename to_sym<T>::type,trans_type>
          ,reduce_type<OP_REDUCE>
          ,symbolic_constant >(trans(t), symbolic_constant("1"));
    }


    template<class ScalarType, class T>\
    typename viennacl::enable_if<is_scalar_expression_t<T>::value ||is_vector_expression_t<T>::value||is_matrix_expression_t<T>::value\
    ,typename convert_to_unary_expr<T,cast_type<ScalarType>,is_vector_expression_t<T>::value\
    ,is_scalar_expression_t<T>::value\
    ,is_matrix_expression_t<T>::value>::type>::type cast(T const & t)\
    {\
      return typename convert_to_unary_expr<T,cast_type<ScalarType>
          ,is_vector_expression_t<T>::value\
          ,is_scalar_expression_t<T>::value\
          ,is_matrix_expression_t<T>::value>::type(make_sym(t));\
    }

/** @brief create an interface to opencl 1 argument functions */
#define MAKE_BUILTIN_FUNCTION1(namefun) \
  template<class T>\
  typename viennacl::enable_if<is_scalar_expression_t<T>::value ||is_vector_expression_t<T>::value||is_matrix_expression_t<T>::value\
  ,typename convert_to_unary_expr<T,namefun##_type,is_vector_expression_t<T>::value\
  ,is_scalar_expression_t<T>::value\
  ,is_matrix_expression_t<T>::value>::type>::type namefun (T const & t)\
    {\
    return typename convert_to_unary_expr<T,namefun##_type\
    ,is_vector_expression_t<T>::value\
    ,is_scalar_expression_t<T>::value\
    ,is_matrix_expression_t<T>::value>::type(make_sym(t));\
  }

/** @brief create an interface to opencl 2 arguments functions */
#define MAKE_BUILTIN_FUNCTION2(namefun) \
  template<class LHS, class RHS>\
  typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)\
  ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)\
  ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)\
  ,typename convert_to_binary_expr<LHS,namefun##_type,RHS\
  ,create_vector<LHS,RHS>::value\
  ,create_scalar<LHS,RHS>::value\
  ,create_matrix<LHS,RHS>::value>::type>::type namefun(LHS const & lhs, RHS const & rhs){\
    return typename convert_to_binary_expr<LHS,namefun##_type,RHS\
    ,create_vector<LHS,RHS>::value\
    ,create_scalar<LHS,RHS>::value\
    ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));\
  }


    MAKE_BUILTIN_FUNCTION1(acos)
    MAKE_BUILTIN_FUNCTION1(acosh)
    MAKE_BUILTIN_FUNCTION1(acospi)
    MAKE_BUILTIN_FUNCTION1(asin)
    MAKE_BUILTIN_FUNCTION1(asinh)
    MAKE_BUILTIN_FUNCTION1(asinpi)
    MAKE_BUILTIN_FUNCTION1(atan)
    MAKE_BUILTIN_FUNCTION2(atan2)
    MAKE_BUILTIN_FUNCTION1(atanh)
    MAKE_BUILTIN_FUNCTION1(atanpi)
    MAKE_BUILTIN_FUNCTION2(atan2pi)
    MAKE_BUILTIN_FUNCTION1(cbrt)
    MAKE_BUILTIN_FUNCTION1(ceil)
    MAKE_BUILTIN_FUNCTION2(copysign)
    MAKE_BUILTIN_FUNCTION1(cos)
    MAKE_BUILTIN_FUNCTION1(cosh)
    MAKE_BUILTIN_FUNCTION1(cospi)
    MAKE_BUILTIN_FUNCTION1(erfc)
    MAKE_BUILTIN_FUNCTION1(erf)
    MAKE_BUILTIN_FUNCTION1(exp)
    MAKE_BUILTIN_FUNCTION1(exp2)
    MAKE_BUILTIN_FUNCTION1(exp10)
    MAKE_BUILTIN_FUNCTION1(expm1)
    MAKE_BUILTIN_FUNCTION1(fabs)
    MAKE_BUILTIN_FUNCTION2(fdim)
    MAKE_BUILTIN_FUNCTION1(floor)
    //MAKE_BUILTIN_FUNCTION3(fma)
    MAKE_BUILTIN_FUNCTION2(fmax)
    MAKE_BUILTIN_FUNCTION2(fmin)
    MAKE_BUILTIN_FUNCTION2(fmod)
    //    MAKE_BUILTIN_FUNCTION1(fract)
    //    MAKE_BUILTIN_FUNCTION1(frexp)
    MAKE_BUILTIN_FUNCTION2(hypot)
    MAKE_BUILTIN_FUNCTION1(ilogb)
    MAKE_BUILTIN_FUNCTION2(ldexp)
    MAKE_BUILTIN_FUNCTION1(lgamma)
    //    MAKE_BUILTIN_FUNCTION1(lgamma_r)
    MAKE_BUILTIN_FUNCTION1(log)
    MAKE_BUILTIN_FUNCTION1(log2)
    MAKE_BUILTIN_FUNCTION1(log10)
    MAKE_BUILTIN_FUNCTION1(log1p)
    MAKE_BUILTIN_FUNCTION1(logb)
    //MAKE_BUILTIN_FUNCTION3(mad)
    //    MAKE_BUILTIN_FUNCTION1(modf)
    MAKE_BUILTIN_FUNCTION1(nan)
    MAKE_BUILTIN_FUNCTION2(nextafter)
    MAKE_BUILTIN_FUNCTION2(pow)
    MAKE_BUILTIN_FUNCTION2(pown)
    MAKE_BUILTIN_FUNCTION2(powr)
    MAKE_BUILTIN_FUNCTION2(remainder)
    //    MAKE_BUILTIN_FUNCTION1(remquo)
    MAKE_BUILTIN_FUNCTION1(rint)
    MAKE_BUILTIN_FUNCTION1(rootn)
    MAKE_BUILTIN_FUNCTION1(round)
    MAKE_BUILTIN_FUNCTION1(rsqrt)
    MAKE_BUILTIN_FUNCTION1(sign)
    MAKE_BUILTIN_FUNCTION1(sin)
    //    MAKE_BUILTIN_FUNCTION1(sincos)
    MAKE_BUILTIN_FUNCTION1(sinh)
    MAKE_BUILTIN_FUNCTION1(sinpi)
    MAKE_BUILTIN_FUNCTION1(sqrt)
    MAKE_BUILTIN_FUNCTION1(tan)
    MAKE_BUILTIN_FUNCTION1(tanh)
    MAKE_BUILTIN_FUNCTION1(tanpi)
    MAKE_BUILTIN_FUNCTION1(tgamma)
    MAKE_BUILTIN_FUNCTION1(trunc)


    //Integer functions
    MAKE_BUILTIN_FUNCTION2(max)
    MAKE_BUILTIN_FUNCTION2(min)

    template<class ScalarType>
    symbolic_vector<ScalarType, vector_handle_accessor,
              binary_vector_expression<
                  binary_vector_expression<
                      unary_vector_expression<binary_vector_expression<cpu_symbolic_scalar<int>,add_type,index_set>, cast_type<int> > , max_type, symbolic_constant
                  >
              ,min_type
            ,cpu_symbolic_scalar<int> > >
    shift(vector<ScalarType> const & t, int k){
      return symbolic_vector<ScalarType, vector_handle_accessor,
          binary_vector_expression<
              binary_vector_expression<
                  unary_vector_expression<binary_vector_expression<cpu_symbolic_scalar<int>,add_type,index_set>, cast_type<int> > , max_type, symbolic_constant
              >
          ,min_type
        ,cpu_symbolic_scalar<int> > >(t.get().size(),
                                        vector_handle_accessor(t.get().handle())
                                        ,generator::min(generator::max(cast<int>(k+index_set()),constant_vector<0>()),(int)t.get().size() - 1));
    }


  }

}


#endif // DUMMY_TYPES_HPP
