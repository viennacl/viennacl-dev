#ifndef VIENNACL_GENERATOR_OPERATORS_HPP
#define VIENNACL_GENERATOR_OPERATORS_HPP

#include <string>

#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl{

namespace generator{


//////////////////////////////////////
///// OPERATOR BASE
//////////////////////////////////////

class op_infos_base {
public:
    op_infos_base(std::string const & name) : name_(name){ }
    std::string const & name() const { return name_; }
protected:
    std::string name_;
};

class binary_op_infos_base : public op_infos_base {
public:
    virtual std::string generate(std::string const & lhs, std::string const & rhs) const = 0;
    binary_op_infos_base(std::string const & name) : op_infos_base(name){ }
};

class binary_fun_op_infos_base : public binary_op_infos_base{
public:
    binary_fun_op_infos_base(std::string const & name) : binary_op_infos_base(name){ }
    std::string generate(std::string const  & lhs, std::string const & rhs) const{
        return name_+"("+lhs+","+rhs+")";
    }
};

class nonarithmetic_op_infos_base : public binary_op_infos_base{
public:
    nonarithmetic_op_infos_base(std::string name) : binary_op_infos_base(name){ }
};

class arithmetic_op_infos_base : public binary_op_infos_base{
public:
    std::string generate(std::string const & lhs, std::string const & rhs) const{
        return lhs + expr_ + rhs;
    }
protected:
    arithmetic_op_infos_base(std::string const & name, std::string const & expr) :  binary_op_infos_base(name), expr_(expr){ }
private:
    std::string expr_;
};

class assignment_op_infos_base : public arithmetic_op_infos_base{
public:
    assignment_op_infos_base(std::string const & name, std::string const & expr) : arithmetic_op_infos_base(name,expr){ }
};

class unary_op_infos_base : public op_infos_base {
public:
    virtual std::string generate(std::string const & sub) const = 0;
    unary_op_infos_base(std::string const & name) : op_infos_base(name) { }
};



class unary_arithmetic_op_infos_base : public unary_op_infos_base{
public:
    std::string generate(std::string const &  sub) const{ return expr_+sub; }
    unary_arithmetic_op_infos_base(std::string const & name, std::string const & expr) : unary_op_infos_base(name), expr_(expr){ }
private:
    std::string expr_;
};

class unary_fun_op_infos_base : public unary_op_infos_base{
public:
    unary_fun_op_infos_base(std::string const & name) : unary_op_infos_base(name){ }
    std::string generate(std::string const & sub) const {
        return name_+"("+sub+")";
    }
};

template<class REDUCE_TYPE>
class reduce_type : public nonarithmetic_op_infos_base{
public:
    reduce_type() : nonarithmetic_op_infos_base("prod"), op_reduce_(new REDUCE_TYPE()){ }
    binary_op_infos_base* op_reduce(){ return op_reduce_.get(); }
    std::string generate(std::string const & lhs, std::string const & rhs) const {
        return op_reduce_->generate(lhs,rhs);
    }

private:
    viennacl::tools::shared_ptr<binary_op_infos_base> op_reduce_;
};

////////////////////////////
//// BINARY AND UNARY OPERATORS
///////////////////////////

#define MAKE_OP(name,expression,base) \
class name##_type : public base{\
    public:\
    name##_type() : base(#name,#expression){ }\
};

MAKE_OP(assign,=,assignment_op_infos_base)
MAKE_OP(inplace_add,+=,assignment_op_infos_base)
MAKE_OP(inplace_sub,-=,assignment_op_infos_base)
MAKE_OP(inplace_scal_mul,*=,assignment_op_infos_base)
MAKE_OP(inplace_scal_div,/=,assignment_op_infos_base)

MAKE_OP(add,+,arithmetic_op_infos_base)
MAKE_OP(sub,-,arithmetic_op_infos_base)

MAKE_OP(unary_sub,-,unary_arithmetic_op_infos_base)
MAKE_OP(identity, ,unary_arithmetic_op_infos_base)

class trans_type : public unary_op_infos_base{
public:
    trans_type() : unary_op_infos_base("trans"){ }
    std::string generate(const std::string &sub) const { return sub; }
};

class replicate_type : public unary_op_infos_base{
public:
    replicate_type() : unary_op_infos_base("replicate"){ }
    std::string generate(const std::string &sub) const { return sub; }
};

class mul_type : public arithmetic_op_infos_base{
public:
    mul_type() : arithmetic_op_infos_base("mul_type","*"){ }
    std::string generate(std::string const & lhs, std::string const & rhs) const{
        if(lhs=="1" && rhs=="1") return "1";
        else if(rhs=="1") return lhs;
        else if(lhs=="1") return rhs;
        else return lhs + "*" + rhs;
    }
};

class div_type : public arithmetic_op_infos_base{
public:
    div_type() : arithmetic_op_infos_base("div_type","/"){ }
    std::string generate(std::string const & lhs, std::string const & rhs) const{
        if(rhs=="1") return lhs;
        else return lhs + "/" + rhs;
    }
};



#undef MAKE_OP

/////////////////////////////////////////
/////////////// BUILTIN FUNCTIONS
/////////////////////////////////////////

#define MAKE_UNARY_FUN_OP(name) \
class name##_type : public unary_fun_op_infos_base{\
    public:\
    name##_type() : unary_fun_op_infos_base(#name){ }\
};

#define MAKE_BINARY_FUN_OP(name) \
class name##_type : public binary_fun_op_infos_base{\
    public:\
    name##_type() : binary_fun_op_infos_base(#name){ }\
};




MAKE_UNARY_FUN_OP(acos)
MAKE_UNARY_FUN_OP(acosh)
MAKE_UNARY_FUN_OP(acospi)
MAKE_UNARY_FUN_OP(asin)
MAKE_UNARY_FUN_OP(asinh)
MAKE_UNARY_FUN_OP(asinpi)
MAKE_UNARY_FUN_OP(atan)
MAKE_BINARY_FUN_OP(atan2)
MAKE_UNARY_FUN_OP(atanh)
MAKE_UNARY_FUN_OP(atanpi)
MAKE_BINARY_FUN_OP(atan2pi)
MAKE_UNARY_FUN_OP(cbrt)
MAKE_UNARY_FUN_OP(ceil)
MAKE_BINARY_FUN_OP(copysign)
MAKE_UNARY_FUN_OP(cos)
MAKE_UNARY_FUN_OP(cosh)
MAKE_UNARY_FUN_OP(cospi)
MAKE_UNARY_FUN_OP(erfc)
MAKE_UNARY_FUN_OP(erf)
MAKE_UNARY_FUN_OP(exp)
MAKE_UNARY_FUN_OP(exp2)
MAKE_UNARY_FUN_OP(exp10)
MAKE_UNARY_FUN_OP(expm1)
MAKE_UNARY_FUN_OP(fabs)
MAKE_BINARY_FUN_OP(fdim)
MAKE_UNARY_FUN_OP(floor)
//MAKE_BUILTIN_FUNCTION3(fma)
MAKE_BINARY_FUN_OP(fmax)
MAKE_BINARY_FUN_OP(fmin)
MAKE_BINARY_FUN_OP(fmod)
//    MAKE_UNARY_FUN_OP(fract)
//    MAKE_UNARY_FUN_OP(frexp)
MAKE_BINARY_FUN_OP(hypot)
MAKE_UNARY_FUN_OP(ilogb)
MAKE_BINARY_FUN_OP(ldexp)
MAKE_UNARY_FUN_OP(lgamma)
//    MAKE_UNARY_FUN_OP(lgamma_r)
MAKE_UNARY_FUN_OP(log)
MAKE_UNARY_FUN_OP(log2)
MAKE_UNARY_FUN_OP(log10)
MAKE_UNARY_FUN_OP(log1p)
MAKE_UNARY_FUN_OP(logb)
//MAKE_BUILTIN_FUNCTION3(mad)
//    MAKE_UNARY_FUN_OP(modf)
MAKE_UNARY_FUN_OP(nan)
MAKE_BINARY_FUN_OP(nextafter)
MAKE_BINARY_FUN_OP(pow)
MAKE_BINARY_FUN_OP(pown)
MAKE_BINARY_FUN_OP(powr)
MAKE_BINARY_FUN_OP(remainder)
//    MAKE_UNARY_FUN_OP(remquo)
MAKE_UNARY_FUN_OP(rint)
MAKE_UNARY_FUN_OP(rootn)
MAKE_UNARY_FUN_OP(round)
MAKE_UNARY_FUN_OP(rsqrt)
MAKE_UNARY_FUN_OP(sin)
//    MAKE_UNARY_FUN_OP(sincos)
MAKE_UNARY_FUN_OP(sinh)
MAKE_UNARY_FUN_OP(sinpi)
MAKE_UNARY_FUN_OP(sqrt)
MAKE_UNARY_FUN_OP(tan)
MAKE_UNARY_FUN_OP(tanh)
MAKE_UNARY_FUN_OP(tanpi)
MAKE_UNARY_FUN_OP(tgamma)
MAKE_UNARY_FUN_OP(trunc)

#undef MAKE_UNARY_FUN_OP
#undef MAKE_BINARY_FUN_OP
}

}
#endif // OPERATORS_HPP
