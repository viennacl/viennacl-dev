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
private:
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
    arithmetic_op_infos_base( std::string const & expr, std::string const & name, bool is_assignment) :  binary_op_infos_base(name,is_assignment), expr_(expr){ }
private:
    std::string expr_;
};

class assign_type : public arithmetic_op_infos_base{
public:
    assign_type() : arithmetic_op_infos_base(" = ", "eq",true){ }
};

class elementwise_prod_type : public arithmetic_op_infos_base{
public:
  elementwise_prod_type() : arithmetic_op_infos_base(" * ", "elementwise_prod",false){ }
};

class elementwise_div_type : public arithmetic_op_infos_base{
public:
    elementwise_div_type() : arithmetic_op_infos_base(" +/ ", "elementwise_div",false){ }
};


class add_type : public arithmetic_op_infos_base{
public:
  add_type() : arithmetic_op_infos_base(" + ", "p",false){ }
};

class inplace_add_type : public arithmetic_op_infos_base{
public:
  inplace_add_type() : arithmetic_op_infos_base(" += ", "p_eq",true){ }
};

class sub_type : public arithmetic_op_infos_base{
public:
  sub_type() : arithmetic_op_infos_base(" - ", "m",false){ }
};

class inplace_sub_type : public arithmetic_op_infos_base{
public:
  inplace_sub_type() : arithmetic_op_infos_base(" -= ", "m_eq",true){ }
};

class scal_mul_type : public arithmetic_op_infos_base{
public:
  scal_mul_type() : arithmetic_op_infos_base(" * ", "mu",false){ }
};

class inplace_scal_mul_type : public arithmetic_op_infos_base{
    inplace_scal_mul_type() : arithmetic_op_infos_base(" *= ", "mu_eq",true){ }
};


class scal_div_type : public arithmetic_op_infos_base{
  scal_div_type() : arithmetic_op_infos_base(" / ", "div", false){ }
};

class inplace_scal_div_type :  public arithmetic_op_infos_base{
    inplace_scal_div_type() : arithmetic_op_infos_base(" /= ", "div_eq", true){ }
};

class inner_prod_type : public nonarithmetic_op_infos_base{
public:
    inner_prod_type() : nonarithmetic_op_infos_base("inprod"){ }
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

class trans_type : public nonarithmetic_op_infos_base{
public:
    trans_type() : nonarithmetic_op_infos_base("trans"){ }
};



class unary_op_infos_base : public op_infos_base{
public:
    std::string generate(std::string const &  sub) const{
        return expr_+sub;
    }
protected:
    unary_op_infos_base(std::string const & name, std::string const & expr) : op_infos_base(name), expr_(expr){ }
private:
    std::string expr_;
};

class unary_sub_type : public unary_op_infos_base{
public:
    unary_sub_type() : unary_op_infos_base("unasub", "-"){ }
};

}

}
#endif // OPERATORS_HPP
