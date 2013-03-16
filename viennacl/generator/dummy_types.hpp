#ifndef VIENNACL_GENERATOR_DUMMY_TYPES_HPP
#define VIENNACL_GENERATOR_DUMMY_TYPES_HPP

#include "viennacl/meta/enable_if.hpp"
#include "viennacl/generator/forwards.h"
//#include "viennacl/forwards.h"
#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/distributed/forwards.hpp"
#include "viennacl/vector.hpp"
#include <set>

namespace viennacl{

namespace generator{

//template<class T, bool deep_copy>
//struct member_storage{
//    typedef T const & type;
//};

//template<class T>
//struct member_storage<T,true>{
//    typedef T type;
//};


//template<class LHS, class OP, class RHS, bool deep_copy=false>
//class binary_tree{
//private:
//    typedef typename member_storage<LHS, deep_copy>::type LhsStorage;
//    typedef typename member_storage<RHS, deep_copy>::type RhsStorage;
//public:
//    typedef LHS Lhs;
//    typedef OP Op;
//    typedef RHS Rhs;
//    LHS const & lhs() const{return lhs_;}
//    RHS const & rhs() const{return rhs_;}
//    OP const & op() const{ return op_; }
//protected:
//    binary_tree(LHS const & lhs, RHS const & rhs) : lhs_(lhs), rhs_(rhs){}
//private:
//    LhsStorage lhs_;
//    RhsStorage rhs_;
//    OP  op_;
//};


//template<class LHS, class OP, class RHS, bool deep_copy=false>
//class binary_vector_expression_wrapper : public binary_tree<LHS,OP,RHS, deep_copy>{
//public: binary_vector_expression_wrapper(LHS const & lhs, RHS const & rhs) : binary_tree<LHS,OP,RHS, deep_copy>(lhs,rhs){ }
//};

//template<class LHS, class OP, class RHS, bool deep_copy=false>
//class binary_scalar_expression_wrapper : public binary_tree<LHS,OP,RHS, deep_copy>{
//public: binary_scalar_expression_wrapper(LHS const & lhs, RHS const & rhs) : binary_tree<LHS,OP,RHS, deep_copy>(lhs,rhs){ }
//};
//template<class LHS, class OP, class RHS,  bool deep_copy=false>
//class binary_matrix_expression_wrapper : public binary_tree<LHS,OP,RHS, deep_copy>{
//public: binary_matrix_expression_wrapper(LHS const & lhs, RHS const & rhs) : binary_tree<LHS,OP,RHS, deep_copy>(lhs,rhs){ }
//};

//template<class LHS, class OP, class RHS, bool deep_copy>
//struct member_storage<binary_matrix_expression_wrapper<LHS,OP,RHS,deep_copy>, false>{
//    typedef binary_matrix_expression_wrapper<LHS,OP,RHS,deep_copy> type;
//};


////////////////////////////////////////

//template<class SUB, class OP, bool deep_copy=false>
//class unary_tree{
//private:
//    typedef typename member_storage<SUB, deep_copy>::type SubStorage;
//public:
//    typedef SUB Sub;
//    typedef OP Op;
//    SUB const & sub() const{return sub_;}
//    OP const & op() const{ return op_; }
//protected:
//    unary_tree(Sub const & sub) : sub_(sub){ }
//private:
//    SubStorage sub_;
//    OP  op_;
//};

//template<class SUB, class OP, bool deep_copy=false>
//class unary_vector_expression_wrapper : public unary_tree<SUB,OP, deep_copy>{
//public: unary_vector_expression_wrapper(SUB const & sub) : unary_tree<SUB,OP,deep_copy>(sub){ }
//};

//template<class SUB, class OP, bool deep_copy=false>
//class unary_scalar_expression_wrapper : public unary_tree<SUB,OP, deep_copy>{
//public: unary_scalar_expression_wrapper(SUB const & sub) : unary_tree<SUB,OP,deep_copy>(sub){ }
//};

//template<class SUB, class OP, bool deep_copy=false>
//class unary_matrix_expression_wrapper : public unary_tree<SUB,OP, deep_copy>{
//public: unary_matrix_expression_wrapper(SUB const & sub) : unary_tree<SUB,OP,deep_copy>(sub){ }
//};

/////////////////////////////////////



//template<class LHS, class RHS, class OP_REDUCE,  bool deep_copy>
//class binary_matrix_expression_wrapper<LHS,prod_type<OP_REDUCE>,RHS, deep_copy> : public binary_tree<LHS,prod_type<OP_REDUCE>,RHS, deep_copy>{
//public:
//    binary_matrix_expression_wrapper(LHS const & lhs, RHS const & rhs, std::string expr = "dot(#1,#2)") : binary_tree<LHS,prod_type<OP_REDUCE>, RHS, deep_copy>(lhs,rhs) ,f_("",expr){ }

//    std::string expr() const { return f_.expr(); }
//private:
//    function_wrapper f_;
//};


//template<class LHS, class RHS, class OP_REDUCE,  bool deep_copy>
//class binary_vector_expression_wrapper<LHS,prod_type<OP_REDUCE>,RHS, deep_copy> : public binary_tree<LHS,prod_type<OP_REDUCE>,RHS, deep_copy>{
//public:
//    binary_vector_expression_wrapper(LHS const & lhs, RHS const & rhs, std::string expr = "dot(#1,#2)") : binary_tree<LHS,prod_type<OP_REDUCE>, RHS, deep_copy>(lhs,rhs) ,f_("",expr){ }

//    std::string expr() const { return f_.expr(); }
//private:
//    function_wrapper f_;
//};


//template<class LHS, class RHS, class OP_REDUCE,  bool deep_copy>
//class binary_scalar_expression_wrapper<LHS,prod_type<OP_REDUCE>,RHS, deep_copy> : public binary_tree<LHS,prod_type<OP_REDUCE>,RHS, deep_copy>{
//public:
//    binary_scalar_expression_wrapper(LHS const & lhs, RHS const & rhs, std::string expr = "dot(#1,#2)") : binary_tree<LHS,prod_type<OP_REDUCE>, RHS, deep_copy>(lhs,rhs) ,f_("",expr){ }

//    std::string expr() const { return f_.expr(); }
//private:
//    function_wrapper f_;
//};


template<class T1, class T2=void, class T3=void>
struct function_wrapper_impl{
    function_wrapper_impl(std::string const & _name, std::string const & _expr) : name(_name), expr(_expr), t1(NULL), t2(NULL), t3(NULL){ }
    std::string name;
    std::string expr;
    T1 const * t1;
    T2 const * t2;
    T3 const * t3;
};




class function_wrapper{
public:
    function_wrapper(std::string const & name
                     ,std::string const & expr) : name_(name), expr_(expr){
        n_args_ = 0;
        bool keep_going = true;
        while(keep_going){
            std::string current_arg = "#"+to_string(n_args_+1);
            if(expr_.find(current_arg)!=std::string::npos)
                ++n_args_;
            else
                keep_going=false;
        }
        assert(n_args_>0 && "\nNo argument specified for the function\n"
                            "\nRecall : 1st arg : #1\n"
                            "\n         2nd arg : #2\n"
                                      "...");
    }

    template<class T1>
    function_wrapper_impl<T1> operator()(T1 const & t1){
        assert(n_args_==1);
        function_wrapper_impl<T1> res(name_,expr_);
        res.t1 = &t1;
        return res;
    }

    template<class T1, class T2>
    function_wrapper_impl<T1,T2> operator()(T1 const & t1, T2 const & t2){
        assert(n_args_==2);
        function_wrapper_impl<T1, T2> res(name_,expr_);
        res.t1 = &t1; res.t2 = &t2;
        return res;
    }

    template<class T1, class T2, class T3>
    function_wrapper_impl<T1,T2, T3> operator()(T1 const & t1, T2 const & t2, T3 const & t3){
        assert(n_args_==3);
        function_wrapper_impl<T1, T2,T3> res(name_,expr_);
        res.t1 = &t1; res.t2 = &t2; res.t3 = &t3;
        return res;
    }

    std::string expr() const { return expr_; }

private:
    std::string name_;
    std::string expr_;
    unsigned int n_args_;
};

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

template<class ScalarType>
struct to_sym<dummy_vector<ScalarType> >{
    typedef symbolic_vector<ScalarType> type;
    static type result(dummy_vector<ScalarType> const & t) { return t.get(); }
};


template<class ScalarType>
class dummy_scalar{
    typedef dummy_scalar<ScalarType> self_type;
    typedef viennacl::scalar<ScalarType> vcl_scal_t;
    vcl_scal_t const & scal_;
public:

    dummy_scalar(vcl_scal_t const & scal): scal_(scal){ }

    vcl_scal_t const & scal() const{ return scal_; }

    template<typename RHS_TYPE>
    binary_scalar_expression<self_type, assign_type, typename to_sym<RHS_TYPE>::type  >
    operator= ( RHS_TYPE const & rhs ){
      return binary_scalar_expression<self_type,assign_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_scalar_expression<self_type, inplace_scal_mul_type, typename to_sym<RHS_TYPE>::type  >
    operator*= ( RHS_TYPE const & rhs ){
      return binary_scalar_expression<self_type,inplace_scal_mul_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_scalar_expression<self_type, inplace_scal_div_type, typename to_sym<RHS_TYPE>::type  >
    operator/= ( RHS_TYPE const & rhs ){
      return binary_scalar_expression<self_type,inplace_scal_div_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_scalar_expression<self_type, inplace_add_type, typename to_sym<RHS_TYPE>::type  >
    operator+= ( RHS_TYPE const & rhs ){
      return binary_scalar_expression<self_type,inplace_add_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_scalar_expression<self_type, inplace_sub_type, typename to_sym<RHS_TYPE>::type  >
    operator-= ( RHS_TYPE const & rhs ){
      return binary_scalar_expression<self_type,inplace_sub_type,RHS_TYPE >(*this,rhs);
    }
};


template<class VCL_MATRIX>
class dummy_matrix{
    typedef dummy_matrix<VCL_MATRIX> self_type;
public:

    typedef VCL_MATRIX gpu_type;

    dummy_matrix(VCL_MATRIX & mat) : mat_(mat){ }

    VCL_MATRIX & mat() const{
        return mat_;
    }

    template<typename RHS_TYPE>
    binary_matrix_expression<self_type, assign_type, typename to_sym<RHS_TYPE>::type  >
    operator= ( RHS_TYPE const & rhs ){
      return binary_matrix_expression<self_type,assign_type,RHS_TYPE >(*this,rhs);
    }

    binary_matrix_expression<self_type, assign_type, self_type>
    operator= ( self_type const & rhs ){
      return binary_matrix_expression<self_type,assign_type, self_type >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_matrix_expression<self_type, inplace_scal_mul_type, typename to_sym<RHS_TYPE>::type  >
    operator*= ( RHS_TYPE const & rhs ){
      return binary_matrix_expression<self_type,inplace_scal_mul_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_matrix_expression<self_type, inplace_scal_div_type, typename to_sym<RHS_TYPE>::type  >
    operator/= ( RHS_TYPE const & rhs ){
      return binary_matrix_expression<self_type,inplace_scal_div_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_matrix_expression<self_type, inplace_add_type, typename to_sym<RHS_TYPE>::type  >
    operator+= ( RHS_TYPE const & rhs ){
      return binary_matrix_expression<self_type,inplace_add_type,RHS_TYPE >(*this,rhs);
    }

    template<typename RHS_TYPE>
    binary_matrix_expression<self_type, inplace_sub_type, typename to_sym<RHS_TYPE>::type  >
    operator-= ( RHS_TYPE const & rhs ){
      return binary_matrix_expression<self_type,inplace_sub_type,RHS_TYPE >(*this,rhs);
    }
private:
    VCL_MATRIX & mat_;
};


template<class T>
struct is_vector_expression_t{ enum { value = 0 }; };
template<typename ScalarType>
struct is_vector_expression_t<dummy_vector<ScalarType> >{ enum { value = 1}; };
template<class LHS, class OP, class RHS>
struct is_vector_expression_t<binary_vector_expression<LHS,OP,RHS> >{ enum { value = 1}; };
template<class SUB, class OP>
struct is_vector_expression_t<unary_vector_expression<SUB,OP> >{ enum { value = 1}; };


template<class T>
struct is_scalar_expression_t{ enum { value = 0 }; };
template<class ScalarType>
struct is_scalar_expression_t<dummy_scalar<ScalarType> >{ enum { value = 1}; };
template<class LHS, class OP, class RHS>
struct is_scalar_expression_t<binary_scalar_expression<LHS,OP,RHS> >{ enum { value = 1}; };
template<class T1, class T2, class T3>
struct is_scalar_expression_t<function_wrapper_impl<T1,T2,T3> >{ enum { value = 1}; };
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
struct convert_to_binary_expr<LHS,OP,RHS,true,false,false>{ typedef binary_vector_expression<LHS,OP,RHS> type; };
template<class LHS, class OP, class RHS>
struct convert_to_binary_expr<LHS,OP,RHS,false,true,false>{ typedef binary_scalar_expression<LHS,OP,RHS> type; };
template<class LHS, class OP, class RHS>
struct convert_to_binary_expr<LHS,OP,RHS,false,false,true>{ typedef binary_matrix_expression<LHS,OP,RHS> type; };



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

//template<class T>
//unary_minus<T> operator -(T const &)
//{
//  return unary_minus<T>();
//}


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


template<class LHS, class RHS>
typename viennacl::enable_if<is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value
                            ,binary_scalar_expression<typename to_sym<LHS>::type,prod_type<add_type>, typename to_sym<RHS>::type> >::type
inner_prod(LHS const & lhs, RHS const & rhs)
{
    return binary_scalar_expression<typename to_sym<LHS>::type,prod_type<add_type>,typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
}

template<class LHS, class RHS>
typename viennacl::enable_if<is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value
                            ,binary_matrix_expression<typename to_sym<LHS>::type,prod_type<add_type>,typename to_sym<RHS>::type> >::type
prod(LHS const & lhs, RHS const & rhs)
{
    return binary_matrix_expression<typename to_sym<LHS>::type,prod_type<add_type>,typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
}


template<class LHS, class RHS>
typename viennacl::enable_if<is_matrix_expression_t<LHS>::value && is_vector_expression_t<RHS>::value
                            ,binary_vector_expression<typename to_sym<LHS>::type,prod_type<add_type>,typename to_sym<RHS>::type> >::type
prod(LHS const & lhs, RHS const & rhs)
{
    return binary_vector_expression<typename to_sym<LHS>::type,prod_type<add_type>,typename to_sym<RHS>::type>(make_sym(lhs),make_sym(rhs));
}


//template<class OP_TYPE, class LHS, class RHS>
//typename viennacl::enable_if<is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value
//                            ,binary_matrix_expression<LHS,prod_type<add_type>,RHS> >::type
//prod_based(LHS const & lhs, RHS const & rhs, std::string const & expression)
//{
//    return binary_matrix_expression<LHS,prod_type<OP_TYPE>,RHS>(lhs,rhs,expression);
//}

template<class T>
typename viennacl::enable_if<is_matrix_expression_t<T>::value, symbolic_matrix<typename T::vcl_t> >::type
trans(T const & mat){
    return make_sym(mat);
}

//template<class LHS, class RHS>
//typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
//                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
//                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
//                            ,typename convert_to_binary_expr<LHS,elementwise_prod_type,RHS
//                                                    ,create_vector<LHS,RHS>::value
//                                                    ,create_scalar<LHS,RHS>::value
//                                                    ,create_matrix<LHS,RHS>::value>::type>::type
//element_prod(LHS const & lhs, RHS const & rhs){
//    return typename convert_to_binary_expr<LHS,elementwise_prod_type,RHS
//            ,create_vector<LHS,RHS>::value
//            ,create_scalar<LHS,RHS>::value
//            ,create_matrix<LHS,RHS>::value>::type(lhs,rhs);
//}

//template<class LHS, class RHS>
//typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
//                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
//                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
//                            ,typename convert_to_binary_expr<LHS,add_type,RHS
//                                                    ,create_vector<LHS,RHS>::value
//                                                    ,create_scalar<LHS,RHS>::value
//                                                    ,create_matrix<LHS,RHS>::value>::type>::type
//operator+(LHS const & lhs, RHS const & rhs){
//    return typename convert_to_binary_expr<LHS,add_type,RHS
//            ,create_vector<LHS,RHS>::value
//            ,create_scalar<LHS,RHS>::value
//            ,create_matrix<LHS,RHS>::value>::type(lhs,rhs);
//}

//template<class LHS, class RHS>
//typename viennacl::enable_if< (is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value)
//                             ||(is_vector_expression_t<LHS>::value && is_vector_expression_t<RHS>::value)
//                             ||(is_matrix_expression_t<LHS>::value && is_matrix_expression_t<RHS>::value)
//                            ,typename convert_to_binary_expr<LHS,sub_type,RHS
//                                                    ,create_vector<LHS,RHS>::value
//                                                    ,create_scalar<LHS,RHS>::value
//                                                    ,create_matrix<LHS,RHS>::value>::type>::type
//operator-(LHS const & lhs, RHS const & rhs){
//    return typename convert_to_binary_expr<LHS,sub_type,RHS
//            ,create_vector<LHS,RHS>::value
//            ,create_scalar<LHS,RHS>::value
//            ,create_matrix<LHS,RHS>::value>::type(lhs,rhs);
//}
//template<class LHS, class RHS>
//typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
//                            ,typename convert_to_binary_expr<LHS,scal_mul_type,RHS
//                            ,create_vector<LHS,RHS>::value
//                            ,create_scalar<LHS,RHS>::value
//                            ,create_matrix<LHS,RHS>::value>::type>::type
//operator*(LHS const & lhs, RHS const & rhs){
//    return typename convert_to_binary_expr<LHS,scal_mul_type,RHS
//                                    ,create_vector<LHS,RHS>::value
//                                    ,create_scalar<LHS,RHS>::value
//                                    ,create_matrix<LHS,RHS>::value>::type(lhs,rhs);
//}

//template<class LHS, class RHS>
//typename viennacl::enable_if< is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
//                            ,typename convert_to_binary_expr<LHS,scal_div_type,RHS
//                                                    ,is_vector_expression_t<LHS>::value || is_vector_expression_t<RHS>::value
//                                                    ,is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
//                                                    ,is_matrix_expression_t<LHS>::value || is_matrix_expression_t<RHS>::value>::type>::type
//operator/(LHS const & lhs, RHS const & rhs){
//    return typename convert_to_binary_expr<LHS,scal_div_type,RHS
//            ,is_vector_expression_t<LHS>::value || is_vector_expression_t<RHS>::value
//            ,is_scalar_expression_t<LHS>::value || is_scalar_expression_t<RHS>::value
//            ,is_matrix_expression_t<LHS>::value || is_matrix_expression_t<RHS>::value>::type(lhs,rhs);
//}


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

/*
template<> static unsigned long get_operation_id<assign_type>(assign_type const &){ return 0; }
template<> static unsigned long get_operation_id<add_type>(add_type const &){ return 1; }


template<> static unsigned long get_operation_id<dummy_vector<float> >(dummy_vector<float> const &){ return 0; }
template<> static unsigned long get_operation_id<dummy_matrix<float> >(dummy_matrix<float> const &){ return 1; }
template<> static unsigned long get_operation_id<dummy_scalar<float> >(dummy_scalar<float> const &){ return 2; }
template<class T1> static unsigned long get_operation_id<function_wrapper_impl<T1> >(function_wrapper_impl<T1> const &){ return 3; }
*/



#define MAKE_BUILTIN_FUNCTION1(name) static function_wrapper name = function_wrapper(#name,#name "(#1)")
#define MAKE_BUILTIN_FUNCTION2(name) static function_wrapper name = function_wrapper(#name,#name "(#1,#2)")
#define MAKE_BUILTIN_FUNCTION3(name) static function_wrapper name = function_wrapper(#name,#name "(#1,#2,#3)")

MAKE_BUILTIN_FUNCTION1(acos);
MAKE_BUILTIN_FUNCTION1(acosh);
MAKE_BUILTIN_FUNCTION1(acospi);
MAKE_BUILTIN_FUNCTION1(asin);
MAKE_BUILTIN_FUNCTION1(asinh);
MAKE_BUILTIN_FUNCTION1(asinpi);
MAKE_BUILTIN_FUNCTION1(atan);
MAKE_BUILTIN_FUNCTION2(atan2);
MAKE_BUILTIN_FUNCTION1(atanh);
MAKE_BUILTIN_FUNCTION1(atanpi);
MAKE_BUILTIN_FUNCTION2(atan2pi);
MAKE_BUILTIN_FUNCTION1(cbrt);
MAKE_BUILTIN_FUNCTION1(ceil);
MAKE_BUILTIN_FUNCTION2(copysign);
MAKE_BUILTIN_FUNCTION1(cos);
MAKE_BUILTIN_FUNCTION1(cosh);
MAKE_BUILTIN_FUNCTION1(cospi);
MAKE_BUILTIN_FUNCTION1(erfc);
MAKE_BUILTIN_FUNCTION1(erf);
MAKE_BUILTIN_FUNCTION1(exp);
MAKE_BUILTIN_FUNCTION1(exp2);
MAKE_BUILTIN_FUNCTION1(exp10);
MAKE_BUILTIN_FUNCTION1(expm1);
MAKE_BUILTIN_FUNCTION1(fabs);
MAKE_BUILTIN_FUNCTION2(fdim);
MAKE_BUILTIN_FUNCTION1(floor);
MAKE_BUILTIN_FUNCTION3(fma);
MAKE_BUILTIN_FUNCTION2(fmax);
MAKE_BUILTIN_FUNCTION2(fmin);
MAKE_BUILTIN_FUNCTION2(fmod);
//    MAKE_BUILTIN_FUNCTION1(fract);
//    MAKE_BUILTIN_FUNCTION1(frexp);
MAKE_BUILTIN_FUNCTION2(hypot);
MAKE_BUILTIN_FUNCTION1(ilogb);
MAKE_BUILTIN_FUNCTION2(ldexp);
MAKE_BUILTIN_FUNCTION1(lgamma);
//    MAKE_BUILTIN_FUNCTION1(lgamma_r);
MAKE_BUILTIN_FUNCTION1(log);
MAKE_BUILTIN_FUNCTION1(log2);
MAKE_BUILTIN_FUNCTION1(log10);
MAKE_BUILTIN_FUNCTION1(log1p);
MAKE_BUILTIN_FUNCTION1(logb);
MAKE_BUILTIN_FUNCTION3(mad);
//    MAKE_BUILTIN_FUNCTION1(modf);
MAKE_BUILTIN_FUNCTION1(nan);
MAKE_BUILTIN_FUNCTION2(nextafter);
MAKE_BUILTIN_FUNCTION2(pow);
MAKE_BUILTIN_FUNCTION2(pown);
MAKE_BUILTIN_FUNCTION2(powr);
MAKE_BUILTIN_FUNCTION2(remainder);
//    MAKE_BUILTIN_FUNCTION1(remquo);
MAKE_BUILTIN_FUNCTION1(rint);
MAKE_BUILTIN_FUNCTION1(rootn);
MAKE_BUILTIN_FUNCTION1(round);
MAKE_BUILTIN_FUNCTION1(rsqrt);
MAKE_BUILTIN_FUNCTION1(sin);
//    MAKE_BUILTIN_FUNCTION1(sincos);
MAKE_BUILTIN_FUNCTION1(sinh);
MAKE_BUILTIN_FUNCTION1(sinpi);
MAKE_BUILTIN_FUNCTION1(sqrt);
MAKE_BUILTIN_FUNCTION1(tan);
MAKE_BUILTIN_FUNCTION1(tanh);
MAKE_BUILTIN_FUNCTION1(tanpi);
MAKE_BUILTIN_FUNCTION1(tgamma);
MAKE_BUILTIN_FUNCTION1(trunc);



}

}


#endif // DUMMY_TYPES_HPP
