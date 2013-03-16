#ifndef VIENNACL_GENERATOR_DUMMY_TYPES_HPP
#define VIENNACL_GENERATOR_DUMMY_TYPES_HPP

#include "viennacl/meta/enable_if.hpp"
#include "viennacl/generator/forwards.h"
//#include "viennacl/forwards.h"
#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/operators.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/distributed/forwards.hpp"
#include "viennacl/vector.hpp"
#include <set>

namespace viennacl{

namespace generator{

template<class T>
struct to_sym{
    typedef T type;
    static type result(T const & t){ return t; }
};

template<class T>
static typename to_sym<T>::type make_sym(T const & t){
    return to_sym<T>::result(t);
}

template<typename SCALARTYPE>
class dummy_vector{
    typedef dummy_vector<SCALARTYPE> self_type;
    typedef viennacl::vector<SCALARTYPE> vcl_t;
public:

    dummy_vector(vcl_t const & vec): vec_(vec){ }

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

template<unsigned int N>
class constant_vector{
public:
    static const unsigned int val = N;
};

template<class ScalarType>
struct to_sym<dummy_vector<ScalarType> >{
    typedef symbolic_vector<ScalarType> type;
    static type result(dummy_vector<ScalarType> const & t) { return t.get(); }
};

template<unsigned int N>
struct to_sym<constant_vector<N> >{
    typedef symbolic_constant<N> type;
    static type result(constant_vector<N> const &) { return type(); }
};


template<class ScalarType>
class dummy_scalar{
    typedef dummy_scalar<ScalarType> self_type;
public:
    typedef viennacl::scalar<ScalarType> vcl_t;

    dummy_scalar(vcl_t const & scal): scal_(scal){ }

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



template<class ScalarType>
struct to_sym<dummy_scalar<ScalarType> >{
    typedef gpu_symbolic_scalar<ScalarType> type;
    static type result(dummy_scalar<ScalarType> const & t) { return t.get(); }
};

template<>
struct to_sym<float>{
    typedef cpu_symbolic_scalar<float> type;
    static type result(float const & t){ return t; }
};

template<class VCL_MATRIX>
class dummy_matrix{
    typedef dummy_matrix<VCL_MATRIX> self_type;
public:

    typedef VCL_MATRIX vcl_t;

    dummy_matrix(VCL_MATRIX & mat) : mat_(mat){ }

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

template<class VCL_MATRIX_T>
struct to_sym<dummy_matrix<VCL_MATRIX_T> >{
    typedef symbolic_matrix<VCL_MATRIX_T> type;
    static type result(dummy_matrix<VCL_MATRIX_T> const & t) { return type(t.get(),false); }
};


template<class T>
struct is_vector_expression_t{ enum { value = 0 }; };
template<typename ScalarType>
struct is_vector_expression_t<dummy_vector<ScalarType> >{ enum { value = 1}; };
template<unsigned int N>
struct is_vector_expression_t<constant_vector<N> >{ enum { value = 1}; };
template<class LHS, class OP, class RHS>
struct is_vector_expression_t<binary_vector_expression<LHS,OP,RHS> >{ enum { value = 1}; };
template<class SUB, class OP>
struct is_vector_expression_t<unary_vector_expression<SUB,OP> >{ enum { value = 1}; };


template<class T>
struct is_scalar_expression_t{ enum { value = 0 }; };
template<class ScalarType>
struct is_scalar_expression_t<dummy_scalar<ScalarType> >{ enum { value = 1}; };
template<>
struct is_scalar_expression_t<float> { enum { value = 1 }; };
template<class LHS, class OP, class RHS>
struct is_scalar_expression_t<binary_scalar_expression<LHS,OP,RHS> >{ enum { value = 1}; };
template<class SUB, class OP>
struct is_scalar_expression_t<unary_scalar_expression<SUB,OP> >{ enum { value = 1}; };


template<class T>
struct is_matrix_expression_t{ enum { value = 0}; };
template<class VCL_MATRIX>
struct is_matrix_expression_t<dummy_matrix<VCL_MATRIX> >{ enum { value = 1}; };
//template<class Scalartype, class F>
//struct is_matrix_expression_t<viennacl::distributed::multi_matrix<Scalartype, F> >{ enum { value = 1}; };
template<class LHS, class OP, class RHS>
struct is_matrix_expression_t<binary_matrix_expression<LHS,OP,RHS> >{ enum { value = 1}; };
template<class T>
struct is_matrix_expression_t<viennacl::distributed::utils::gpu_wrapper<T> > { enum { value = 1 }; };
template<class SUB, class OP>
struct is_matrix_expression_t<unary_matrix_expression<SUB,OP> >{ enum { value = 1}; };


template<class LHS, class OP, class RHS, bool create_vector, bool create_scalar, bool create_matrix>
struct convert_to_binary_expr;
template<class LHS, class OP, class RHS>
struct convert_to_binary_expr<LHS,OP,RHS,true,false,false>{ typedef binary_vector_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };
template<class LHS, class OP, class RHS>
struct convert_to_binary_expr<LHS,OP,RHS,false,true,false>{ typedef binary_scalar_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };
template<class LHS, class OP, class RHS>
struct convert_to_binary_expr<LHS,OP,RHS,false,false,true>{ typedef binary_matrix_expression<typename to_sym<LHS>::type, OP, typename to_sym<RHS>::type> type; };



template<class T>
struct is_operator{ enum{ value = 0}; };
template<> struct is_operator<assign_type>{ enum { value = 1}; };
template<> struct is_operator<add_type>{ enum { value = 1}; };
template<> struct is_operator<inplace_add_type>{ enum { value = 1}; };
template<> struct is_operator<sub_type>{ enum { value = 1}; };
template<> struct is_operator<inplace_sub_type>{ enum { value = 1}; };
template<> struct is_operator<scal_mul_type>{ enum { value = 1}; };
template<> struct is_operator<inplace_scal_mul_type>{ enum { value = 1}; };
template<> struct is_operator<scal_div_type>{ enum { value = 1}; };
template<> struct is_operator<inplace_scal_div_type>{ enum { value = 1}; };

template<class T>
struct is_leaf{ enum{ value = 0}; };
template<class ScalarType> struct is_leaf<dummy_vector<ScalarType> >{ enum { value = 1 }; };
template<class ScalarType> struct is_leaf<dummy_scalar<ScalarType> >{ enum { value = 1 }; };
template<class VCL_MATRIX> struct is_leaf<dummy_matrix<VCL_MATRIX> >{ enum { value = 1 }; };


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

template<class T>
typename viennacl::enable_if<is_matrix_expression_t<T>::value, symbolic_matrix<typename T::vcl_t> >::type
trans(T const & mat){
    return make_sym(mat);
}

template<class LHS, class RHS>
typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
                            ,typename convert_to_binary_expr<LHS,scal_mul_type,RHS
                                                    ,create_vector<LHS,RHS>::value
                                                    ,create_scalar<LHS,RHS>::value
                                                    ,create_matrix<LHS,RHS>::value>::type>::type
element_prod(LHS const & lhs, RHS const & rhs){
    return typename convert_to_binary_expr<LHS,scal_mul_type,RHS
            ,create_vector<LHS,RHS>::value
            ,create_scalar<LHS,RHS>::value
            ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
}

template<class LHS, class RHS>
typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
                            ,typename convert_to_binary_expr<LHS,add_type,RHS
                                                    ,create_vector<LHS,RHS>::value
                                                    ,create_scalar<LHS,RHS>::value
                                                    ,create_matrix<LHS,RHS>::value>::type>::type
operator+(LHS const & lhs, RHS const & rhs){
    return typename convert_to_binary_expr<LHS,add_type,RHS
            ,create_vector<LHS,RHS>::value
            ,create_scalar<LHS,RHS>::value
            ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
}

template<class LHS, class RHS>
typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
                            ,typename convert_to_binary_expr<LHS,sub_type,RHS
                                                    ,create_vector<LHS,RHS>::value
                                                    ,create_scalar<LHS,RHS>::value
                                                    ,create_matrix<LHS,RHS>::value>::type>::type
operator-(LHS const & lhs, RHS const & rhs){
    return typename convert_to_binary_expr<LHS,sub_type,RHS
            ,create_vector<LHS,RHS>::value
            ,create_scalar<LHS,RHS>::value
            ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
}
template<class LHS, class RHS>
typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
                            ,typename convert_to_binary_expr<LHS,scal_mul_type,RHS
                            ,create_vector<LHS,RHS>::value
                            ,create_scalar<LHS,RHS>::value
                            ,create_matrix<LHS,RHS>::value>::type>::type
operator*(LHS const & lhs, RHS const & rhs){
    return typename convert_to_binary_expr<LHS,scal_mul_type,RHS
                                    ,create_vector<LHS,RHS>::value
                                    ,create_scalar<LHS,RHS>::value
                                    ,create_matrix<LHS,RHS>::value>::type(make_sym(lhs),make_sym(rhs));
}

template<class LHS, class RHS>
typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
                            ,typename convert_to_binary_expr<LHS,scal_div_type,RHS
                            ,create_vector<LHS,RHS>::value
                            ,create_scalar<LHS,RHS>::value
                            ,create_matrix<LHS,RHS>::value>::type>::type
operator/(LHS const & lhs, RHS const & rhs){
    return typename convert_to_binary_expr<LHS,scal_div_type,RHS
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
                            ,binary_scalar_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant<1> > >::type
reduce(T const & t){
    return binary_scalar_expression<typename to_sym<T>::type,reduce_type<OP_REDUCE>,symbolic_constant<1> >(make_sym(t), symbolic_constant<1>());
}





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



}

}


#endif // DUMMY_TYPES_HPP
