#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP

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


/** @file viennacl/generator/symbolic_types.hpp
    @brief Defines the symbolic types for the expressions used by the code generator
*/

#include "viennacl/ocl/utils.hpp"

#include "viennacl/generator/forwards.h"

#include "viennacl/generator/templates/profile_base.hpp"

#include "viennacl/generator/utils.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/backend/memory.hpp"


#include <map>
#include <set>
#include <list>

namespace viennacl{

  namespace generator{


    //////////////////////////////////////
    ///// OPERATOR BASE
    //////////////////////////////////////

    /** @brief base class for operator */
    class operator_base {
      public:
        operator_base(std::string const & name) : name_(name){ }
        std::string const & name() const { return name_; }
        virtual ~operator_base(){ }
      protected:
        std::string name_;
    };

    /** @brief base class for binary operator */
    class binary_operator : public operator_base {
      public:
        virtual std::string generate(std::string const & lhs, std::string const & rhs) const = 0;
        binary_operator(std::string const & name) : operator_base(name){ }
    };

    /** @brief base class for binary function
     *
     *  Implemented as a binary operator
     */
    class symbolic_binary_fun : public binary_operator{
      public:
        symbolic_binary_fun(std::string const & name) : binary_operator(name){ }
        std::string generate(std::string const  & lhs, std::string const & rhs) const{
          return name_+"("+lhs+","+rhs+")";
        }
    };

    /** @brief base class for binary arithmetic operators
     *
     *  Refers to all built-in C99 operators (+, -, &, ||, ...)
     */
    class binary_builtin_operator : public binary_operator{
      public:
        std::string generate(std::string const & lhs, std::string const & rhs) const{
          return lhs + expr_ + rhs;
        }
      protected:
        binary_builtin_operator(std::string const & name, std::string const & expr) :  binary_operator(name), expr_(expr){ }
      private:
        std::string expr_;
    };

    /** @brief base class for assignment operators
     *
     * Refers to =, +=, -=, *=, /=
     */
    class assignment_operator : public binary_builtin_operator{
      public:
        assignment_operator(std::string const & name, std::string const & expr) : binary_builtin_operator(name,expr){ }
    };

    /** @brief base class for unary operators */
    class unary_operator : public operator_base {
      public:
        virtual std::string generate(std::string const & sub) const = 0;
        unary_operator(std::string const & name) : operator_base(name) { }
    };

    /** @brief base class for casting operators
     *
     * \tparam ScalarType destination's scalartype
     */
    template<class ScalarType>
    class cast_type : public unary_operator{
      public:
        cast_type() : unary_operator("cast_"+std::string(utils::print_type<ScalarType>::value())){ }
        std::string generate(const std::string &sub) const { return "("+std::string(utils::print_type<ScalarType>::value())+")(" + sub + ")"; }
    };

    /** @brief base class for unary arithmetic operators
     *
     * Include unary -,+
     */
    class unary_arithmetic_operator : public unary_operator{
      public:
        std::string generate(std::string const &  sub) const{ return expr_+sub; }
        unary_arithmetic_operator(std::string const & name, std::string const & expr) : unary_operator(name), expr_(expr){ }
      private:
        std::string expr_;
    };

    /** @brief base class for unary functions*/
    class symbolic_unary_fun : public unary_operator{
      public:
        symbolic_unary_fun(std::string const & name) : unary_operator(name){ }
        std::string generate(std::string const & sub) const {
          return name_+"("+sub+")";
        }
    };

    /** @brief class for reduction operators
    *
    * \tparam REDUCE_TYPE underlying reduction type. Has to be a binary operator.
    */
    template<class REDUCE_TYPE>
    class reduce_type : public binary_operator{
      public:
        reduce_type() : binary_operator("prod"), op_reduce_(new REDUCE_TYPE()){ }
        binary_operator* op_reduce(){ return op_reduce_.get(); }
        std::string generate(std::string const & lhs, std::string const & rhs) const {
          return op_reduce_->generate(lhs,rhs);
        }

      private:
        viennacl::tools::shared_ptr<binary_operator> op_reduce_;
    };

    ////////////////////////////
    //// BUILTIN OPERATORS
    ///////////////////////////

#define MAKE_OP(name,expression,base) \
  class name##_type : public base{\
  public:\
  name##_type() : base(#name,#expression){ }\
  };

    //Assignment
    MAKE_OP(assign,=,assignment_operator)
    MAKE_OP(inplace_add,+=,assignment_operator)
    MAKE_OP(inplace_sub,-=,assignment_operator)
    MAKE_OP(inplace_scal_mul,*=,assignment_operator)
    MAKE_OP(inplace_scal_div,/=,assignment_operator)

    //Arithmetic
    MAKE_OP(add,+,binary_builtin_operator)
    MAKE_OP(sub,-,binary_builtin_operator)


    //Comparison
    MAKE_OP(sup,>,binary_builtin_operator)
    MAKE_OP(supeq,>=,binary_builtin_operator)
    MAKE_OP(inf,<,binary_builtin_operator)
    MAKE_OP(infeq,<=,binary_builtin_operator)
    MAKE_OP(eqto,==,binary_builtin_operator)
    MAKE_OP(neqto,!=,binary_builtin_operator)

    //Bitwise
    MAKE_OP(and,&,binary_builtin_operator)
    MAKE_OP(or,|,binary_builtin_operator)
    MAKE_OP(xor,^,binary_builtin_operator)



    MAKE_OP(unary_sub,-,unary_arithmetic_operator)
    MAKE_OP(identity, ,unary_arithmetic_operator)

    /** @brief transposition type */
    class trans_type : public unary_operator{
                                public:
                                trans_type() : unary_operator("trans"){ }
                                std::string generate(const std::string &sub) const { return sub; }
  };

    /** @brief multiplication type
     *
     * Accounts for both scalar multiplication and elementwise products
     */
    class mul_type : public binary_builtin_operator{
      public:
        mul_type() : binary_builtin_operator("mul_type","*"){ }
        std::string generate(std::string const & lhs, std::string const & rhs) const{
          if(lhs=="1" && rhs=="1") return "1";
          else if(rhs=="1") return lhs;
          else if(lhs=="1") return rhs;
          else return lhs + "*" + rhs;
        }
    };

    /** @brief multiplication type
     *
     * Accounts for both scalar division and elementwise divisions
     */
    class div_type : public binary_builtin_operator{
      public:
        div_type() : binary_builtin_operator("div_type","/"){ }
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
  class name##_type : public symbolic_unary_fun{\
  public:\
  name##_type() : symbolic_unary_fun(#name){ }\
  };

#define MAKE_BINARY_FUN_OP(name) \
  class name##_type : public symbolic_binary_fun{\
  public:\
  name##_type() : symbolic_binary_fun(#name){ }\
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
    MAKE_UNARY_FUN_OP(sign)
    //    MAKE_UNARY_FUN_OP(sincos)
    MAKE_UNARY_FUN_OP(sinh)
    MAKE_UNARY_FUN_OP(sinpi)
    MAKE_UNARY_FUN_OP(sqrt)
    MAKE_UNARY_FUN_OP(tan)
    MAKE_UNARY_FUN_OP(tanh)
    MAKE_UNARY_FUN_OP(tanpi)
    MAKE_UNARY_FUN_OP(tgamma)
    MAKE_UNARY_FUN_OP(trunc)


    //Integer functions
    MAKE_BINARY_FUN_OP(max)
    MAKE_BINARY_FUN_OP(min)

#undef MAKE_UNARY_FUN_OP
#undef MAKE_BINARY_FUN_OP

    template<class ScalarType>
    struct symbolic_value_argument{
      private:
        typedef typename viennacl::result_of::cl_type<ScalarType>::type cl_type;
      public:
        static std::string generate_header(const char * address_space, unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> &) {
          std::string name = "arg"+utils::to_string(n_arg++);
          kss << address_space << " " << utils::print_type<ScalarType>::value() << " " << name << ",";
          return name;
        }
        static void enqueue(cl_type value, unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & ) {
          k.arg(n_arg++,value);
        }
    };


    template<class ScalarType>
    struct symbolic_pointer_argument{
        static void enqueue(viennacl::backend::mem_handle const & handle, unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued) {
          if(already_enqueued.insert(handle).second==true){
            k.arg(n_arg++,handle.opencl_handle());
          }
        }

        static std::string generate_header(viennacl::backend::mem_handle const & handle, const char * address_space, size_t vector,
                                    unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          if(already_enqueued.insert(std::make_pair(handle,n_arg)).second==true){
            std::string name = "arg"+utils::to_string(n_arg++);
            kss << address_space << ' ' << utils::print_type<ScalarType>::value() ;
            if(vector>1)
              kss << vector;
            kss << "* " << name << ",";
          }
          return "arg"+utils::to_string(already_enqueued.at(handle));
        }

    };

    /** @brief symbolic structure for local memory
         *
         *  A convenience layer to create and access local memory when generating a kernel source code
         *
         * \tparam dim Dimension of the local memory (considered internally as a 1D array for efficiency)
         */
    template<unsigned int dim>
    class symbolic_local_memory;

    template<>
    class symbolic_local_memory<1>{
      public:
        symbolic_local_memory(std::string const & name, unsigned int size, std::string const & scalartype): name_(name), size_(size), scalartype_(scalartype){ }
        std::string declare() const{ return "__local " + scalartype_ + " " + name_ + '[' + utils::to_string(size_) + ']'; }
        unsigned int size() const{ return size_; }
        std::string const & name() const{ return name_; }
        std::string access(std::string const & index) const{ return name_ + '[' + index + ']'; }
      private:
        std::string name_;
        unsigned int size_;
        std::string scalartype_;
    };

    template<>
    class symbolic_local_memory<2>{
      public:
        symbolic_local_memory(std::string const & name, unsigned int size1, unsigned int size2, std::string const & scalartype): size1_(size1), size2_(size2), impl_(name,size1*size2,scalartype){ }
        std::string declare() const{ return impl_.declare(); }
        std::string const & name() const { return impl_.name(); }
        unsigned int size1() const{ return size1_; }
        unsigned int size2() const{ return size2_; }
        std::string access(std::string const & i, std::string const & j) const{ return name() + "[" + '('+i+')' + '*' + utils::to_string(size2_) + "+ (" + j + ") ]";}
      private:
        unsigned int size1_;
        unsigned int size2_;
        symbolic_local_memory<1> impl_;
    };

    /** @brief structure shared by all the symbolic structures associated with the same handle
         *
         *  In for example op.add(x = y + z); op.add(y = x + z);, ensures that the two symbolic vectors created with x, y and z share the same state
         */
    struct shared_symbolic_infos_t{
      public:
        shared_symbolic_infos_t(unsigned int _id, std::string const & _scalartype, unsigned int _scalartype_size, unsigned int _alignment = 1) {
          id = _id;
          scalartype = _scalartype;
          scalartype_size = _scalartype_size;
          alignment = _alignment;
        }
        std::map<unsigned int,std::string> private_values;
        unsigned int id;
        std::string name;
        std::string scalartype;
        unsigned int scalartype_size;
        unsigned int alignment;
    };


    /** @brief Base class for an expression tree */
    class symbolic_expression_tree_base{
      public:
        virtual std::string generate(unsigned int i, int vector_element = -1) const { return ""; }
        virtual void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > >  & shared_infos, code_generation::profile_base const & prof)= 0;
        virtual void access_index(unsigned int i, std::string const & ind0, std::string const & ind1) = 0;
        virtual void fetch(unsigned int i, utils::kernel_generation_stream & kss) = 0;
        virtual void write_back(unsigned int i, utils::kernel_generation_stream & kss) = 0;
        virtual void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued)  = 0;
        virtual void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const = 0;
        virtual void clear_private_value(unsigned int i) = 0;
        virtual bool operator==(symbolic_expression_tree_base const & other) const = 0;
        virtual ~symbolic_expression_tree_base(){ }
        symbolic_expression_tree_base() { }
    };



    /** @brief Base class for binary expression trees */
    class symbolic_binary_expression_tree_infos_base : public symbolic_expression_tree_base{
      public:
        symbolic_expression_tree_base & lhs() const{ return *lhs_; }
        symbolic_expression_tree_base & rhs() const{ return *rhs_; }
        binary_operator & op() { return *op_; }

        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > >  & shared_infos, code_generation::profile_base const & prof){
          lhs_->bind(shared_infos,prof);
          rhs_->bind(shared_infos,prof);
        }

        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
          lhs_->access_index(i,ind0,ind1);
          rhs_->access_index(i,ind0,ind1);
        }

        void fetch(unsigned int i, utils::kernel_generation_stream & kss){
          lhs_->fetch(i,kss);
          rhs_->fetch(i,kss);
        }

        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
           lhs_->generate_header(n_arg, kss, already_enqueued);
           rhs_->generate_header(n_arg, kss, already_enqueued);
        }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          lhs_->enqueue(n_arg, k, already_enqueued, prof);
          rhs_->enqueue(n_arg, k, already_enqueued, prof);
        }

        void clear_private_value(unsigned int i){
          lhs_->clear_private_value(i);
          rhs_->clear_private_value(i);
        }

        void write_back(unsigned int i, utils::kernel_generation_stream & kss){
          if(dynamic_cast<assignment_operator*>(op_.get())) lhs_->write_back(i,kss);
        }

        bool operator==(symbolic_expression_tree_base const & other) const{
          if(symbolic_binary_expression_tree_infos_base const * p = dynamic_cast<symbolic_binary_expression_tree_infos_base const *>(&other)){
            return *lhs_==*p->lhs_ && op_->name()==p->op_->name() && *rhs_==*p->rhs_;
          }
          return false;
        }

        std::string generate(unsigned int i, int vector_element = -1) const {
          return "(" +  op_->generate(lhs_->generate(i,vector_element), rhs_->generate(i,vector_element) ) + ")";
        }

      protected:
        symbolic_binary_expression_tree_infos_base(symbolic_expression_tree_base * lhs, binary_operator * op, symbolic_expression_tree_base * rhs) : lhs_(lhs), op_(op), rhs_(rhs){        }
        viennacl::tools::shared_ptr<symbolic_expression_tree_base> lhs_;
        viennacl::tools::shared_ptr<binary_operator> op_;
        viennacl::tools::shared_ptr<symbolic_expression_tree_base> rhs_;
    };


    /** @brief Base class for binary vector expressions
         *
         *  Here to allow RTTI to retrieve vector expressions
         */
    class symbolic_binary_vector_expression_base : public symbolic_binary_expression_tree_infos_base{
      public:
        symbolic_binary_vector_expression_base( symbolic_expression_tree_base * lhs, binary_operator* op, symbolic_expression_tree_base * rhs) : symbolic_binary_expression_tree_infos_base( lhs,op,rhs){ }
    };

    /** @brief Base class for binary scalar expressions
         *
         *  Here to allow RTTI to retrieve scalar expressions
         */
    class symbolic_binary_scalar_expression_base : public symbolic_binary_expression_tree_infos_base{
      public:
        symbolic_binary_scalar_expression_base( symbolic_expression_tree_base * lhs, binary_operator* op, symbolic_expression_tree_base * rhs) : symbolic_binary_expression_tree_infos_base( lhs,op,rhs){ }
    };

    /** @brief Base class for binary matrix expressions
         *
         *  Here to allow RTTI to retrieve matrix expressions
         */
    class symbolic_binary_matrix_expression_base : public symbolic_binary_expression_tree_infos_base{
      public:
        symbolic_binary_matrix_expression_base( symbolic_expression_tree_base * lhs, binary_operator* op, symbolic_expression_tree_base * rhs) : symbolic_binary_expression_tree_infos_base( lhs,op,rhs){ }
    };


    /** @brief Base class for unary expression trees */
    class symbolic_unary_tree_infos_base : public virtual symbolic_expression_tree_base{
      public:
        symbolic_unary_tree_infos_base(symbolic_expression_tree_base * sub, unary_operator * op) : sub_(sub), op_(op) { }

        symbolic_expression_tree_base & sub() const{ return *sub_; }

        unary_operator const & op() const{ return *op_; }

        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > >  & shared_infos, code_generation::profile_base const & prof){
          sub_->bind(shared_infos,prof);
        }

        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
          if(dynamic_cast<trans_type *>(op_.get())) sub_->access_index(i,ind1,ind0);
          else  sub_->access_index(i,ind0,ind1);
        }

        void fetch(unsigned int i, utils::kernel_generation_stream & kss){ sub_->fetch(i,kss); }

        void write_back(unsigned int i, utils::kernel_generation_stream & kss){}


        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          return sub_->generate_header(n_arg, kss, already_enqueued);
        }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          return sub_->enqueue(n_arg, k, already_enqueued, prof);
        }

        void clear_private_value(unsigned int i){
          sub_->clear_private_value(i);
        }

        std::string generate(unsigned int i, int vector_element = -1) const { return "(" +  op_->generate(sub_->generate(i,vector_element)) + ")"; }

        bool operator==(symbolic_expression_tree_base const & other) const{
          if(symbolic_unary_tree_infos_base const * p = dynamic_cast<symbolic_unary_tree_infos_base const *>(&other)){
            return *sub_==*p->sub_ && op_->name()==p->op_->name();
          }
          return false;
        }
      protected:
        viennacl::tools::shared_ptr<symbolic_expression_tree_base> sub_;
        viennacl::tools::shared_ptr<unary_operator> op_;
    };



    /** @brief Base class for unary vector expressions
         *
         *  Here to allow RTTI to retrieve vector expressions
         */
    class symbolic_unary_vector_expression_base : public symbolic_unary_tree_infos_base{
      public:
        symbolic_unary_vector_expression_base( symbolic_expression_tree_base * sub, unary_operator* op) : symbolic_unary_tree_infos_base( sub,op){ }
    };

    /** @brief Base class for unary scalar expressions
         *
         *  Here to allow RTTI to retrieve scalar expressions
         */
    class symbolic_unary_scalar_expression_base : public symbolic_unary_tree_infos_base{
      public:
        symbolic_unary_scalar_expression_base( symbolic_expression_tree_base * sub, unary_operator* op) : symbolic_unary_tree_infos_base( sub,op){ }
    };

    /** @brief Base class for unary matrix expressions
         *
         *  Here to allow RTTI to retrieve matrix expressions
         */
    class symbolic_unary_matrix_expression_base : public symbolic_unary_tree_infos_base{
      public:
        symbolic_unary_matrix_expression_base( symbolic_expression_tree_base * sub, unary_operator* op) : symbolic_unary_tree_infos_base( sub,op){ }
    };

    /** @brief Base class for symbolic_vector, symbolic_matrix, ... */
    class symbolic_datastructure : public symbolic_expression_tree_base{
      public:
        void private_value(unsigned int i, std::string const & new_name) { infos_->private_values[i] = new_name; }

        void clear_private_value(unsigned int i) { infos_->private_values[i] = ""; }

        std::string const & scalartype() const { return infos_->scalartype; }

        unsigned int scalartype_size() const { return infos_->scalartype_size; }

        std::string aligned_scalartype() const {
          unsigned int alignment = infos_->alignment;
          std::string const & scalartype = infos_->scalartype;
          if(alignment==1){
            return scalartype;
          }
          else{
            assert( (alignment==2 || alignment==4 || alignment==8 || alignment==16) && "Invalid alignment");
            return scalartype + utils::to_string(alignment);
          }
        }

        unsigned int alignment() const { return infos_->alignment; }

        void alignment(unsigned int val) { infos_->alignment = val; }

        virtual ~symbolic_datastructure(){ }
      protected:
        shared_symbolic_infos_t* infos_;
    };

    /** @brief Base class for symbolic scalars passed by value */
    class symbolic_value_scalar_base : public symbolic_datastructure{
      public:
        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
        std::string generate(unsigned int i, int vector_element = -1) const { return infos_->name; }
        void fetch(unsigned int i, utils::kernel_generation_stream & kss){ }
        void write_back(unsigned int i, utils::kernel_generation_stream & kss){ }
    };

    /** @brief Base class for symbolic datastructures passed by pointer */
    class symbolic_pointed_datastructure : public symbolic_datastructure{
      protected:
        virtual std::string access_buffer(unsigned int i) const = 0;
      public:
        void fetch(unsigned int i, utils::kernel_generation_stream & kss){
          if(infos_->private_values[i].empty()){
            std::string val = "private" + utils::to_string(infos_->id) + utils::to_string(i);
            std::string aligned_scalartype = infos_->scalartype;
            if(infos_->alignment > 1) aligned_scalartype += utils::to_string(infos_->alignment);
            kss << aligned_scalartype << " " << val << " = " << access_buffer(i) << ";" << std::endl;
            infos_->private_values[i] = val;
          }
        }
        std::string name() const{
          return infos_->name;
        }
        virtual void write_back(unsigned int i, utils::kernel_generation_stream & kss){
          kss << access_buffer(i) << " = " << infos_->private_values[i] << ";" << std::endl;
          infos_->private_values[i].clear();
        }
        std::string generate(unsigned int i, int vector_element = -1) const {
          std::string res;
          if(infos_->private_values[i].empty()) res = access_buffer(i);
          else res = infos_->private_values[i];
          if(vector_element >= 0 && infos_->alignment > 1) res += ".s" + utils::to_string(vector_element);
          return res;
        }
    };



    /** @brief Base class for symbolic scalars passed by pointer */
    class symbolic_pointer_scalar_base : public symbolic_pointed_datastructure{
      public:
        std::string generate(unsigned int i, int vector_element = -1) const {
          if(infos_->private_values[i].empty()) return "*"+infos_->name;
          else  return infos_->private_values[i];
        }
        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
    };

    /** @brief Base class for symbolic vectors */
    class symbolic_vector_base : public symbolic_pointed_datastructure{
      public:
        symbolic_vector_base(size_t size) : size_(size) { }
        std::string size() const{ return size_name; }
        size_t real_size() const { return size_; }
        virtual ~symbolic_vector_base(){ }
      protected:
        size_t size_;
        std::string size_name;
    };

    /** @brief Base class for symbolic matrices */
    class symbolic_matrix_base : public symbolic_pointed_datastructure{
      public:
        symbolic_matrix_base(bool is_rowmajor) : is_rowmajor_(is_rowmajor){ }
        std::string  internal_size1() const{ return size1_name; }
        std::string  internal_size2() const{ return size2_name; }
        virtual size_t real_size1() const = 0;
        virtual size_t real_size2() const = 0;
        bool const is_rowmajor() const { return is_rowmajor_; }
        std::string offset(std::string const & offset_i, std::string const & offset_j) const {
          if(is_rowmajor_){
            return '(' + offset_i + ')' + '*' + internal_size2() + "+ (" + offset_j + ')';
          }
          return '(' + offset_i + ')' + "+ (" + offset_j + ')' + '*' + internal_size1();
        }

      protected:
        bool is_rowmajor_;
        std::string size1_name;
        std::string size2_name;
    };

    /** @brief Base class for symbolic matrix matrix products */
    class symbolic_matrix_matrix_product_base : public symbolic_binary_matrix_expression_base{
      public:
        symbolic_matrix_matrix_product_base( symbolic_expression_tree_base * lhs, binary_operator* op, symbolic_expression_tree_base * rhs) :
          symbolic_binary_matrix_expression_base(lhs,op,rhs){
        }

        void set_val_name(std::string const & val_name) { val_name_ = val_name; }

        std::string val_name(unsigned int m, unsigned int n){ return val_name_ +  '_' + utils::to_string(m) + '_' + utils::to_string(n); }

        std::string update_val(std::string const & res, std::string const & lhs, std::string const & rhs){
          return res + " = " + op_->generate(res , lhs + "*" + rhs);
        }
      private:
        std::string val_name_;
    };


    /** @brief Base class for symbolic matrix vector products */
    class vector_reduction_base : public symbolic_binary_vector_expression_base{
      public:
        vector_reduction_base( symbolic_expression_tree_base * lhs, binary_operator* op, symbolic_expression_tree_base * rhs) : symbolic_binary_vector_expression_base(lhs,new mul_type,rhs), op_reduce_(op){            }
        binary_operator const & op_reduce() const { return *op_reduce_; }
        void access_name(std::string const & str) { access_name_ = str; }
        std::string generate(unsigned int i, int vector_element = -1) const{ return access_name_; }
      private:
        viennacl::tools::shared_ptr<binary_operator> op_reduce_;
        std::string access_name_;
    };

    /** @brief Base class for symbolic inner products */
    class symbolic_scalar_reduction_base : public symbolic_binary_scalar_expression_base {
      public:
        symbolic_scalar_reduction_base(symbolic_expression_tree_base * lhs, binary_operator * op, symbolic_expression_tree_base * rhs): symbolic_binary_scalar_expression_base(lhs,new mul_type,rhs), op_reduce_(op){ }
        virtual const char * scalartype() const = 0;
        binary_operator const & op_reduce() const { return *op_reduce_; }
        void access_name(std::string const & str) { access_name_ = str; }
        virtual symbolic_local_memory<1> make_local_memory(size_t size)=0;
        virtual std::string sum_name()=0;
        virtual std::string access(std::string const & str)=0;
        virtual std::string name() const = 0;
      protected:
        viennacl::tools::shared_ptr<binary_operator> op_reduce_;
        std::string access_name_;
    };


    template<class T, class Pred>
    static void extract_as(symbolic_expression_tree_base* root, std::list<T*> & args, Pred pred){
      if(symbolic_binary_expression_tree_infos_base* p = dynamic_cast<symbolic_binary_expression_tree_infos_base*>(root)){
        extract_as(&p->lhs(), args,pred);
        extract_as(&p->rhs(),args,pred);
      }
      else if(symbolic_unary_tree_infos_base* p = dynamic_cast<symbolic_unary_tree_infos_base*>(root)){
        extract_as(&p->sub(), args,pred);
      }
      if(T* t = dynamic_cast<T*>(root)){
        if(pred(t)) args.push_back(t);
      }
    }

    template<class T, class Pred>
    static void extract_as(tools::shared_ptr<symbolic_expression_tree_base> const & root, std::list<T*> & args, Pred pred){
      extract_as(root.get(), args, pred);
    }

    template<class T, class Pred>
    static void extract_as_unique(symbolic_expression_tree_base* root, std::list<T*> & args, Pred pred){
      if(symbolic_binary_expression_tree_infos_base* p = dynamic_cast<symbolic_binary_expression_tree_infos_base*>(root)){
        extract_as(&p->lhs(), args,pred);
        extract_as(&p->rhs(),args,pred);
      }
      else if(symbolic_unary_tree_infos_base* p = dynamic_cast<symbolic_unary_tree_infos_base*>(root)){
        extract_as(&p->sub(), args,pred);
      }
      if(T* t = dynamic_cast<T*>(root)){
        if(pred(t)) utils::unique_push_back(args,t);
      }
    }

    template<class T>
    bool is_transposed(T const * t){
      if(symbolic_unary_matrix_expression_base const * m = dynamic_cast<symbolic_unary_matrix_expression_base const *>(t)){
        return static_cast<bool>(dynamic_cast<trans_type const *>(&m->op()));
      }
      if(symbolic_unary_vector_expression_base const * v = dynamic_cast<symbolic_unary_vector_expression_base const *>(t))
        return static_cast<bool>(dynamic_cast<trans_type const *>(&v->op()));
      return false;
    }


    namespace utils{
      struct EXTRACT_IF{
          typedef std::list<symbolic_expression_tree_base*> result_type_single;
          typedef std::list<symbolic_expression_tree_base*> result_type_all;
          static void do_on_new_res(result_type_single & new_res, result_type_single & res){
            res.merge(new_res);
          }
          static void do_on_pred_true(symbolic_expression_tree_base* tree,result_type_single & res){
            res.push_back(tree);
          }
          static void do_on_next_operation_merge(result_type_single & new_res, result_type_all & final_res){
            final_res.merge(new_res);
          }
      };


      template<class FILTER_T, class Pred>
      static typename FILTER_T::result_type_single filter(symbolic_expression_tree_base* const tree, Pred pred){
        typedef typename FILTER_T::result_type_single res_t;
        res_t res;
        if(symbolic_binary_expression_tree_infos_base * p = dynamic_cast<symbolic_binary_expression_tree_infos_base *>(tree)){
          res_t  reslhs(filter<FILTER_T,Pred>(&p->lhs(),pred));
          res_t resrhs(filter<FILTER_T,Pred>(&p->rhs(),pred));
          FILTER_T::do_on_new_res(reslhs,res);
          FILTER_T::do_on_new_res(resrhs,res);
        }
        else if(symbolic_unary_tree_infos_base * p = dynamic_cast<symbolic_unary_tree_infos_base *>(tree)){
          res_t  ressub(filter<FILTER_T,Pred>(&p->sub(),pred));
          FILTER_T::do_on_new_res(ressub,res);
        }
        if(pred(tree)){
          FILTER_T::do_on_pred_true(tree,res);
        }
        return res;
      }

      template<class FILTER_T, class Pred>
      static typename FILTER_T::result_type_all filter(std::list<symbolic_expression_tree_base*> const & trees, Pred pred){
        typedef typename FILTER_T::result_type_single res_t_single;
        typedef typename FILTER_T::result_type_all res_t_all;
        res_t_all res;
        for(std::list<symbolic_expression_tree_base*>::const_iterator it = trees.begin() ; it != trees.end() ; ++it){
          res_t_single tmp(filter<FILTER_T,Pred>(*it,pred));

          FILTER_T::do_on_next_operation_merge(tmp,res);
        }

        return res;
      }


      template<class T, class B>
      static std::list<T *> cast(std::list<B *> const & in){
        std::list<T*> res;
        for(typename std::list<B *>::const_iterator it = in.begin(); it!= in.end() ; ++it){
          if(T* p = dynamic_cast<T*>(*it)){
            res.push_back(p);
          }
        }
        return res;
      }

      template<class T>
      static std::list<T *> extract_cast(std::list<symbolic_expression_tree_base*> const & trees){
        return cast<T,symbolic_expression_tree_base>(filter<EXTRACT_IF>(trees,is_type<T>()));
      }
    }

    /** @brief first_letter of a given scalartype
         *
         * \tparam T Scalar type.
         */
    template<class T> struct first_letter_of;
    template<> struct first_letter_of<float>{ static const std::string value(){ return "f"; } };
    template<> struct first_letter_of<double>{ static const std::string value(){ return "d"; } };
    template<> struct first_letter_of<int>{ static const std::string value(){ return "i"; } };
    template<> struct first_letter_of<long int>{ static const std::string value(){ return "li"; } };
    template<> struct first_letter_of<unsigned int>{ static const std::string value(){ return "ui"; } };
    template<> struct first_letter_of<long unsigned int>{ static const std::string value(){ return "lui"; } };


    template<class T, class Enable=void>
    struct to_sym{
        typedef T type;
        static type result(T const & t){ return t; }
    };

    template<class T>
    static typename to_sym<T>::type make_sym(T const & t){
      return to_sym<T>::result(t);
    }

    /** @brief symbolic binary vector expression
         *
         * \tparam LHS Left-hand side of the expression
         * \tparam OP Operator of the expression
         * \tparam RHS Right-hand side of the expression
         */
    template<class LHS, class OP, class RHS>
    class binary_vector_expression : public symbolic_binary_vector_expression_base{
      public:
        typedef typename LHS::ScalarType ScalarType;
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_vector_expression(LHS const & lhs, RHS const & rhs) :symbolic_binary_vector_expression_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
    };


    /** @brief reduction of each row of a matrix
         *
         * \tparam OP_REDUCE corresponding reduction operator
         */
    template<class LHS, class RHS, class OP_REDUCE>
    class binary_vector_expression<LHS,reduce_type<OP_REDUCE>,RHS> : public vector_reduction_base{
      public:
        typedef typename LHS::ScalarType ScalarType;
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_vector_expression(LHS const & lhs, RHS const & rhs) : vector_reduction_base(new LHS(lhs), new reduce_type<OP_REDUCE>(), new RHS(rhs)){ }
    };


    /** @brief symbolic binary scalar expression  */
    template<class LHS, class OP, class RHS>
    class binary_scalar_expression : public symbolic_binary_scalar_expression_base{
      public:
        typedef typename LHS::ScalarType ScalarType;
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_scalar_expression(LHS const & lhs, RHS const & rhs) :symbolic_binary_scalar_expression_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
    };


    /** @brief symbolic binary matrix expression  */
    template<class LHS, class OP, class RHS>
    class binary_matrix_expression : public symbolic_binary_matrix_expression_base{
      public:
        typedef typename LHS::ScalarType ScalarType;
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_matrix_expression(LHS const & lhs, RHS const & rhs) :symbolic_binary_matrix_expression_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
    };

    /** @brief matrix-matrix product like operations
         *
         * \tparam OP_REDUCE accumulation operator for each element
         */
    template<class LHS, class RHS, class OP_REDUCE>
    class binary_matrix_expression<LHS,reduce_type<OP_REDUCE>,RHS> : public symbolic_matrix_matrix_product_base{
      public:
        typedef typename LHS::ScalarType ScalarType;
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_matrix_expression(LHS const & lhs, RHS const & rhs) : symbolic_matrix_matrix_product_base(new LHS(lhs), new reduce_type<OP_REDUCE>(), new RHS(rhs)){ }
      private:

    };

    /** @brief Unary vector expression */
    template<class UNDERLYING, class OP>
    class unary_vector_expression : public symbolic_unary_vector_expression_base{
      public:
        typedef typename UNDERLYING::ScalarType ScalarType;
        typedef UNDERLYING Underlying;
        unary_vector_expression(UNDERLYING const & underlying) :symbolic_unary_vector_expression_base(new UNDERLYING(underlying), new OP()){ }
    };

    /** @brief Unary scalar expression */
    template<class UNDERLYING, class OP>
    class unary_scalar_expression : public symbolic_unary_scalar_expression_base{
      public:
        typedef typename UNDERLYING::ScalarType ScalarType;
        typedef UNDERLYING Underlying;
        unary_scalar_expression(UNDERLYING const & underlying) :symbolic_unary_scalar_expression_base(new UNDERLYING(underlying), new OP()){ }
    };

    /** @brief Unary matrix expression */
    template<class UNDERLYING, class OP>
    class unary_matrix_expression : public symbolic_unary_matrix_expression_base{
      public:
        typedef typename UNDERLYING::ScalarType ScalarType;
        typedef UNDERLYING Underlying;
        unary_matrix_expression(UNDERLYING const & underlying) :symbolic_unary_matrix_expression_base(new UNDERLYING(underlying), new OP()){ }
    };

    /** @brief Static layer for handling the temporary vectors associated with inner products
         *
         * \tparam ScalarType scalartype of the temporaries
         */
    template<class ScalarType>
    struct inner_product_tempories{
        static std::map<cl_context, viennacl::vector<ScalarType> > map;
    };

    template<class ScalarType>
    std::map<cl_context, viennacl::vector<ScalarType> > inner_product_tempories<ScalarType>::map;









    /**
      * @brief Symbolic scalar type. Will be passed by value.
      *
      * @tparam SCALARTYPE The Scalartype of the scalar in the generated code
      */
    template <typename SCALARTYPE>
    class cpu_symbolic_scalar : public symbolic_value_scalar_base
    {
      public:
        typedef SCALARTYPE ScalarType;
        cpu_symbolic_scalar(ScalarType const & val) :  val_(val){}
        void repr(std::ostringstream & oss) const{
          oss << "vscal" << first_letter_of<SCALARTYPE>::value();
        }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos, code_generation::profile_base const &prof){
          infos_= utils::unique_insert(shared_infos,std::make_pair((symbolic_datastructure *)this,
                                                                   tools::shared_ptr<shared_symbolic_infos_t>(new shared_symbolic_infos_t(shared_infos.size(),utils::print_type<ScalarType>::value(),sizeof(ScalarType)))))->second.get();
        }

        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          infos_->name = symbolic_value_argument<ScalarType>::generate_header("",  n_arg, kss, already_enqueued);
        }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          symbolic_value_argument<ScalarType>::enqueue(val_, n_arg, k, already_enqueued);
        }

        bool operator==(symbolic_expression_tree_base const & other) const{
          if(cpu_symbolic_scalar const * p = dynamic_cast<cpu_symbolic_scalar const *>(&other)) return val_ == p->val_;
          return false;
        }
        std::string generate(unsigned int i, int vector_element = -1) const{  return infos_->name; }
      private:
        ScalarType val_;
    };

    /** @brief Symbolic constant
       *
       * Hardcoded string in the generated kernel
       */
    class symbolic_constant : public symbolic_expression_tree_base{
      public:
        symbolic_constant(std::string const & expr) : expr_(expr){ }
        std::string generate(unsigned int i, int vector_element = -1) const { return expr_; }
        virtual void clear_private_value(unsigned int i){ }
        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
        void fetch(unsigned int i, utils::kernel_generation_stream & kss){ }
        void write_back(unsigned int i, utils::kernel_generation_stream & kss){ }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > >& , code_generation::profile_base const &){ }
        bool operator==(symbolic_expression_tree_base const & other) const{ return dynamic_cast<symbolic_constant const *>(&other); }
        void generate_header(unsigned int &, std::ostringstream &, std::map<viennacl::backend::mem_handle, unsigned int> &) { }
        void enqueue(unsigned int &, viennacl::ocl::kernel &, std::set<viennacl::backend::mem_handle> &, code_generation::profile_base const &) const { }
      private:
        std::string expr_;
    };


    /**
      * @brief Symbolic scalar type. Passed by pointer.
      *
      * @tparam ID The argument ID of the scalar in the generated code
      * @tparam SCALARTYPE The SCALARTYPE of the scalar in the generated code
      */
    template <typename SCALARTYPE>
    class gpu_symbolic_scalar : public symbolic_pointer_scalar_base
    {
      private:
        typedef gpu_symbolic_scalar<SCALARTYPE> self_type;
        std::string access_buffer(unsigned int i) const { return "*"+infos_->name;  }
      public:
        typedef viennacl::scalar<SCALARTYPE> vcl_t;
        typedef SCALARTYPE ScalarType;
        gpu_symbolic_scalar(vcl_t const & vcl_scal) : vcl_scal_(vcl_scal){ }
        void const * handle() const{ return static_cast<void const *>(&vcl_scal_.handle()); }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos, code_generation::profile_base const & prof){
          infos_= utils::unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this,
                                                                    tools::shared_ptr<shared_symbolic_infos_t>(new shared_symbolic_infos_t(shared_infos.size(),utils::print_type<ScalarType>::value(),sizeof(ScalarType)))))->second.get();
        }
        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          infos_->name = symbolic_pointer_argument<ScalarType>::generate_header(vcl_scal_.handle(), "__global ", 1, n_arg, kss, already_enqueued);
        }
        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          symbolic_pointer_argument<ScalarType>::enqueue(vcl_scal_.handle(), n_arg, k, already_enqueued);
        }
        bool operator==(symbolic_expression_tree_base const & other) const{
          if(gpu_symbolic_scalar const * p = dynamic_cast<gpu_symbolic_scalar const *>(&other)){
            return vcl_scal_.handle().opencl_handle() == p->vcl_scal_.handle().opencl_handle();
          }
          return false;
        }
        void repr(std::ostringstream & oss) const{
          oss << "pscal" << first_letter_of<SCALARTYPE>::value();
        }
      private:
        vcl_t const & vcl_scal_;
    };







    /**
        * @brief Symbolic matrix type
        *
        * @tparam VCL_MATRIX underlying wished viennacl matrix type
        * @tparam ELEMENTS_ACCESSOR policy to access the elements (they do not necessarily come from memory, but can also be constant for example!)
        * @tparem ROW_INDEX policy to access a rows (using operations on index_set())
        * @tparam COL_INDEX policy to access a column (using operations on index_set())
        */
    template<class VCL_MATRIX, class ELEMENT_ACCESSOR, class ROW_INDEX, class COL_INDEX>
    class symbolic_matrix : public symbolic_matrix_base
    {
        typedef symbolic_matrix<VCL_MATRIX, ELEMENT_ACCESSOR, ROW_INDEX, COL_INDEX> self_type;
        std::string access_buffer(unsigned int i) const {
          return elements_.access(row_index_.generate(i), col_index_.generate(i), size1_name, size2_name, is_rowmajor_);
        }
      public:
        typedef VCL_MATRIX vcl_t;
        typedef typename vcl_t::value_type::value_type ScalarType;

        symbolic_matrix(size_t size1, size_t size2, ELEMENT_ACCESSOR const & elements, ROW_INDEX const & row_accessor, COL_INDEX const & col_index) : symbolic_matrix_base(utils::are_same_type<typename VCL_MATRIX::orientation_category,viennacl::row_major_tag>::value)
        , size1_(size1), size2_(size2), elements_(elements), row_index_(row_accessor), col_index_(col_index){ }

        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos, code_generation::profile_base const & prof){
          infos_= utils::unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this,
                                                                    tools::shared_ptr<shared_symbolic_infos_t>(new shared_symbolic_infos_t(shared_infos.size(),utils::print_type<ScalarType>::value(),sizeof(ScalarType), prof.vectorization()))))->second.get();

          elements_.bind(shared_infos, prof, infos_);
        }

        ELEMENT_ACCESSOR const & elements() const { return elements_; }
        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
            infos_->name = elements_.template generate_header<ScalarType>(n_arg,kss,already_enqueued);
            row_index_.generate_header(n_arg, kss, already_enqueued);
            col_index_.generate_header(n_arg, kss, already_enqueued);
            size1_name = symbolic_value_argument<unsigned int>::generate_header("", n_arg, kss, already_enqueued);
            size2_name = symbolic_value_argument<unsigned int>::generate_header("", n_arg, kss, already_enqueued);
        }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
            elements_.template enqueue<ScalarType>(n_arg,k,already_enqueued,prof);
            row_index_.enqueue(n_arg, k, already_enqueued, prof);
            col_index_.enqueue(n_arg, k, already_enqueued, prof);
            cl_uint size1_arg = cl_uint(size1_);
            cl_uint size2_arg = cl_uint(size2_);
            if(is_rowmajor_)
              size2_arg /= prof.vectorization();
            else
              size1_arg /= prof.vectorization();
            symbolic_value_argument<unsigned int>::enqueue(size1_arg,n_arg,k,already_enqueued);
            symbolic_value_argument<unsigned int>::enqueue(size2_arg,n_arg,k,already_enqueued);
        }

        bool operator==(symbolic_expression_tree_base const & other) const{
          if(symbolic_matrix const * p = dynamic_cast<symbolic_matrix const *>(&other))
            return typeid(other)==typeid(*this)
                &&row_index_==p->row_index_
                &&col_index_==p->col_index_
                &&elements_==p->elements_;
          return false;
        }

        size_t real_size1() const { return size1_; }

        size_t real_size2() const { return size2_; }
        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
          row_index_.access_index(i, ind0, "0");
          col_index_.access_index(i, ind1, "0");
        }
      protected:
        size_t size1_;
        size_t size2_;
        ELEMENT_ACCESSOR elements_;
        ROW_INDEX row_index_;
        COL_INDEX col_index_;
    };




    /**
        * @brief Symbolic matrix type
        *
        * @tparam SCALARTYPE scalartype of the symbolic vector
        * @tparam ELEMENTS_ACCESSOR policy to access the elements (they do not necessarily come from memory, but can also be constant for example!)
        * @tparem INDEX policy to access the vector (using operations on index_set())
    */
    template <class SCALARTYPE, class ELEMENT_ACCESSOR, class INDEX>
    class symbolic_vector : public symbolic_vector_base{
      private:
        std::string access_buffer(unsigned int i) const { return elements_.access(index_.generate(i)); }
      public:
        typedef SCALARTYPE ScalarType;
        symbolic_vector(size_t size, ELEMENT_ACCESSOR const & elements, INDEX const & accessor) : symbolic_vector_base(size), elements_(elements), index_(accessor){ }

        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos, code_generation::profile_base const & prof){
          infos_= utils::unique_insert(shared_infos,std::make_pair((symbolic_datastructure *)this, tools::shared_ptr<shared_symbolic_infos_t>(new shared_symbolic_infos_t(shared_infos.size(),utils::print_type<ScalarType>::value(),sizeof(ScalarType), prof.vectorization()))))->second.get();
          index_.bind(shared_infos,prof);
          elements_.bind(shared_infos,prof,infos_);
        }

        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          infos_->name = elements_.template generate_header<ScalarType>(n_arg,kss,already_enqueued);
          index_.generate_header(n_arg,kss,already_enqueued);
          size_name = symbolic_value_argument<unsigned int>::generate_header("", n_arg, kss, already_enqueued);
        }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          elements_.template enqueue<ScalarType>(n_arg, k, already_enqueued, prof);
          index_.enqueue(n_arg,k,already_enqueued, prof);
          symbolic_value_argument<unsigned int>::enqueue(size_, n_arg, k, already_enqueued);
        }

        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
          assert(ind1=="0");
          index_.access_index(i, ind0, "0");
        }

        bool operator==(symbolic_expression_tree_base const & other) const{
          if(symbolic_vector const * p = dynamic_cast<symbolic_vector const *>(&other))
            return elements_==p->elements_ && index_==p->index_;
          return false;
        }
      protected:
        ELEMENT_ACCESSOR elements_;
        INDEX index_;
    };


    /** @brief Set of indexes for the generated kernel
     *
     *  For example, symbolic_vector(index_set) corresponds to regular accesses, while symbolic_vector(index_set + 1) translates all the access indexes by 1 to the right
    */
    class index_set : public symbolic_expression_tree_base{
      public:
        index_set() { }
        void access_index(unsigned int i, std::string const & ind0, std::string const &){ ind0s_[i] = ind0; }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > >  & shared_infos, code_generation::profile_base const & prof){ }
        void fetch(unsigned int i, utils::kernel_generation_stream & kss){ }
        void write_back(unsigned int i, utils::kernel_generation_stream & kss){ }
        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> &) { }
        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const { }
        void clear_private_value(unsigned int i){ }
        std::string generate(unsigned int i, int vector_element = -1) const { return ind0s_.at(i); }
        bool operator==(symbolic_expression_tree_base const & other) const{ return dynamic_cast<index_set const *>(&other); }
      private:
        std::map<unsigned int, std::string> ind0s_;
    };

    /**
     * @brief The handle_element_accessor class
     *
     * The most common/straightforward element accessor, accesses a memory buffer at a given index.
     */
    class handle_element_accessor{
      public:
        handle_element_accessor(viennacl::backend::mem_handle const & h) : h_(h){ }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos
                  ,code_generation::profile_base const & prof
                  ,shared_symbolic_infos_t const * infos){ infos_ = infos; }

        template<class ScalarType>
        std::string generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          return symbolic_pointer_argument<ScalarType>::generate_header(h_, "__global", infos_->alignment, n_arg, kss, already_enqueued);
        }

        template<class ScalarType>
        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          return symbolic_pointer_argument<ScalarType>::enqueue(h_, n_arg, k, already_enqueued);
        }

        bool operator==(handle_element_accessor const & other) const{
          return h_ == other.h_;
        }

      protected:
        shared_symbolic_infos_t const * infos_;
        viennacl::backend::mem_handle const & h_;
    };

    class matrix_handle_accessor : public handle_element_accessor{
      public:
        matrix_handle_accessor(viennacl::backend::mem_handle const & h) : handle_element_accessor(h){ }
        std::string access(std::string const & ind1, std::string const & ind2, std::string const & size1, std::string const & size2, bool is_rowmajor) const{
          std::string ind;
          if(is_rowmajor){
            ind = '(' + ind1 + ')' + '*' + size2 + "+ (" + ind2 + ')';
          }
          else{
            ind = '(' + ind1 + ')' + "+ (" + ind2 + ')' + '*' + size1;
          }
          return infos_->name + "[" + ind + "]";
        }
    };

    template<class MatrixT>
    class matrix_repmat_accessor : public handle_element_accessor {
      public:
        matrix_repmat_accessor(size_t size1, size_t size2, viennacl::backend::mem_handle const & h) : handle_element_accessor(h), underlying_(size1,size2,matrix_handle_accessor(h),index_set(), index_set()){

        }

        std::string access(std::string const & ind1, std::string const & ind2, std::string const & size1, std::string const & size2, bool is_rowmajor) const{
          return underlying_.elements().access(ind1 + "%" + underlying_.internal_size1(), ind2 + "%" + underlying_.internal_size2(), underlying_.internal_size1(), underlying_.internal_size2(), underlying_.is_rowmajor());
        }

        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos
                  ,code_generation::profile_base const & prof
                  ,shared_symbolic_infos_t const * infos){
          underlying_.bind(shared_infos, prof);
        }

        template<class ScalarType>
        std::string generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued)  {
           underlying_.generate_header(n_arg, kss, already_enqueued);
           return underlying_.name();
        }

        template<class ScalarType>
        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
           underlying_.enqueue(n_arg, k, already_enqueued, prof);
        }
      private:
        symbolic_matrix<MatrixT, matrix_handle_accessor, index_set, index_set> underlying_;
    };

    /**
     * @brief The matrix_diag_accessor class
     *
     * The handle should refer to a vector, element(i,j) = (i==j)?vec[i]:0;
     */
    class matrix_diag_accessor : public handle_element_accessor{
      public:
        matrix_diag_accessor(viennacl::backend::mem_handle const & h) : handle_element_accessor(h){ }
        std::string access(std::string const & ind1, std::string const & ind2, std::string size1, std::string size2, bool is_rowmajor) const{
          return '('+ind1+"=="+ind2+")?"+infos_->name+"["+ind1+"]:0";
        }
    };


    class vector_handle_accessor : public handle_element_accessor{
      public:
        vector_handle_accessor(viennacl::backend::mem_handle const & h) : handle_element_accessor(h){ }
        std::string access(std::string const & ind) const{ return infos_->name + "[" + ind + "]"; }
    };


    /** @brief Reduction from a vector to a scalar
         *
         * \tparam Reduction operator
         */
    template<class LHS, class OP_REDUCE, class RHS>
    class binary_scalar_expression<LHS, reduce_type<OP_REDUCE>, RHS > : public symbolic_scalar_reduction_base{
        typedef typename LHS::ScalarType ScalarType;
      public:
        typedef LHS Lhs;
        typedef RHS Rhs;
        binary_scalar_expression(LHS const & lhs, RHS const & rhs):  symbolic_scalar_reduction_base(new LHS(lhs), new OP_REDUCE, new RHS(rhs))
                                                     ,tmp_(1024,vector_handle_accessor(inner_product_tempories<ScalarType>::map[viennacl::ocl::current_context().handle().get()].handle()),index_set()){
          inner_product_tempories<ScalarType>::map[viennacl::ocl::current_context().handle().get()].resize(1024);
        }
        void generate_header(unsigned int & n_arg, std::ostringstream & kss, std::map<viennacl::backend::mem_handle, unsigned int> & already_enqueued) {
          symbolic_binary_scalar_expression_base::generate_header(n_arg, kss, already_enqueued);
          tmp_.generate_header(n_arg,kss,already_enqueued);
        }
        const char * scalartype() const {
          return utils::print_type<ScalarType>::value();
        }
        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k, std::set<viennacl::backend::mem_handle> & already_enqueued, code_generation::profile_base const & prof) const{
          symbolic_binary_scalar_expression_base::enqueue(n_arg,k,already_enqueued,prof);
          tmp_.enqueue(n_arg,k,already_enqueued,prof);
        }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_symbolic_infos_t> > > & shared_infos, code_generation::profile_base const & prof){
          lhs_->bind(shared_infos,prof);
          rhs_->bind(shared_infos,prof);
          tmp_.bind(shared_infos,prof);
        }
        std::string generate(unsigned int i, int vector_element = -1) const{
            return access_name_;
        }
        std::string name() const{
            return tmp_.name();
        }

        symbolic_local_memory<1> make_local_memory(size_t size){
          return symbolic_local_memory<1>(tmp_.name()+"_local", size, utils::print_type<ScalarType>::value());
        }
        std::string sum_name(){
          return tmp_.name() + "_reduced";
        }
        std::string access(const std::string &str){
          return tmp_.name() + "[" + str + "]";
        }

        bool operator==(symbolic_expression_tree_base const & other) const{
          return false;
        }

      private:
        symbolic_vector<ScalarType, vector_handle_accessor, index_set> tmp_;
    };

  }

}
#endif // SYMBOLIC_TYPES_BASE_HPP
