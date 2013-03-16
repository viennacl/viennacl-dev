#ifndef VIENNACL_GENERATOR_OPERATORS_HPP
#define VIENNACL_GENERATOR_OPERATORS_HPP

#include <string>

#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl{

namespace generator{

class op_infos_base {
public:
    op_infos_base(std::string const & name) : name_(name){ }
    std::string const & name() const { return name_; }
protected:
    std::string name_;
};

class binary_op_infos_base : public op_infos_base {
public:
    bool is_assignment() const { return is_assignment_; }
    virtual std::string generate(std::string const & lhs, std::string const & rhs) const = 0;
    binary_op_infos_base(std::string const & name, bool is_assignment) : op_infos_base(name), is_assignment_(is_assignment){ }
private:
    bool is_assignment_;
};

class binary_fun_op_infos_base : public binary_op_infos_base{
public:
    std::string generate(std::string const  & lhs, std::string const & rhs) const{
        return name_+"("+lhs+","+rhs+")";
    }
};

class nonarithmetic_op_infos_base : public binary_op_infos_base{
public:
    nonarithmetic_op_infos_base(std::string name) : binary_op_infos_base(name,false){ }
};

class arithmetic_op_infos_base : public binary_op_infos_base{
public:
    std::string generate(std::string const & lhs, std::string const & rhs) const{
        return lhs + expr_ + rhs;
    }
protected:
    arithmetic_op_infos_base(std::string const & name, std::string const & expr, bool is_assignment) :  binary_op_infos_base(name,is_assignment), expr_(expr){ }
private:
    std::string expr_;
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
class prod_type : public nonarithmetic_op_infos_base{
public:
    prod_type() : nonarithmetic_op_infos_base("prod"), op_reduce_(new REDUCE_TYPE()){ }
    binary_op_infos_base* op_reduce(){ return op_reduce_.get(); }
    std::string generate(std::string const & lhs, std::string const & rhs) const {
        return op_reduce_->generate(lhs,rhs);
    }

private:
    viennacl::tools::shared_ptr<binary_op_infos_base> op_reduce_;
};

#define MAKE_BINARY_ARITHMETIC_OP(name,expression,is_assignment) \
class name##_type : public arithmetic_op_infos_base{\
    public:\
    name##_type() : arithmetic_op_infos_base(#name,#expression,is_assignment){ }\
};

MAKE_BINARY_ARITHMETIC_OP(assign,=,true)
MAKE_BINARY_ARITHMETIC_OP(inplace_add,+=,true)
MAKE_BINARY_ARITHMETIC_OP(inplace_sub,-=,true)
MAKE_BINARY_ARITHMETIC_OP(inplace_scal_mul,*=,true)
MAKE_BINARY_ARITHMETIC_OP(inplace_scal_div,/=,true)

MAKE_BINARY_ARITHMETIC_OP(add,+,false)
MAKE_BINARY_ARITHMETIC_OP(sub,-,false)
MAKE_BINARY_ARITHMETIC_OP(scal_mul,*,false)
MAKE_BINARY_ARITHMETIC_OP(scal_div,/,false)
MAKE_BINARY_ARITHMETIC_OP(elementwise_prod,*,false)
MAKE_BINARY_ARITHMETIC_OP(elementwise_div,/,false)
#undef MAKE_BINARY_ARITHMETIC_OP



#define MAKE_UNARY_ARITHMETIC_OP(name,expression) \
class name##_type : public unary_arithmetic_op_infos_base{\
    public:\
    name##_type() : unary_arithmetic_op_infos_base(#name,#expression){ }\
};

MAKE_UNARY_ARITHMETIC_OP(unary_sub,-)
#undef MAKE_UNARY_ARITHMETIC_OP


#define MAKE_UNARY_FUN_OP(name) \
class name##_type : public unary_fun_op_infos_base{\
    public:\
    name##_type() : unary_fun_op_infos_base(#name){ }\
};

MAKE_UNARY_FUN_OP(exp)
#undef MAKE_UNARY_FUN_OP



}

}
#endif // OPERATORS_HPP
