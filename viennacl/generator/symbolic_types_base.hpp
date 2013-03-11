#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP


#include "viennacl/forwards.h"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/backend/mem_handle.hpp"

#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/forwards.h"

#include <map>
#include <set>
#include <list>

namespace viennacl{

    namespace generator{

        template<unsigned int dim>
        class local_memory;

        template<>
        class local_memory<1>{
        public:

            local_memory(std::string const & name, unsigned int size, std::string const & scalartype): name_(name), size_(size), scalartype_(scalartype){ }

            std::string declare() const{
                return "__local " + scalartype_ + " " + name_ + '[' + to_string(size_) + ']';
            }

            unsigned int size() const{ return size_; }

            std::string const & name() const{
                return name_;
            }

            std::string access(std::string const & index) const{
                return name_ + '[' + index + ']';
            }

        private:
            std::string name_;
            unsigned int size_;
            std::string const & scalartype_;
        };

        template<>
        class local_memory<2>{
        public:

            local_memory(std::string const & name
                         , unsigned int size1
                         , unsigned int size2
                         , std::string const & scalartype): size1_(size1), size2_(size2), impl_(name,size1*size2,scalartype){

            }

            std::string declare() const{
                return impl_.declare();
            }

            std::string const & name() const { return impl_.name(); }

            unsigned int size1() const{ return size1_; }

            unsigned int size2() const{ return size2_; }

            std::string offset(std::string const & i, std::string const & j) const{
                return '('+i+')' + '*' + to_string(size2_) + '(' + j + ')';
            }

        private:
            unsigned int size1_;
            unsigned int size2_;
            local_memory<1> impl_;
        };

        struct shared_infos_t{
        public:
            shared_infos_t(unsigned int id, std::string scalartype, unsigned int scalartype_size, unsigned int alignment = 1) : id_(id), name_("arg"+to_string(id)), scalartype_(scalartype), scalartype_size_(scalartype_size),alignment_(alignment){ }
            std::string const & access_name(unsigned int i){ return access_names_.at(i); }
            void  access_name(unsigned int i, std::string const & name_){ access_names_[i] = name_; }
            std::string const & name() const{ return name_; }
            std::string const & scalartype() const{ return scalartype_; }
            unsigned int const & scalartype_size() const{ return scalartype_size_; }
            unsigned int alignment() const{ return alignment_; }
            void alignment(unsigned int val) { alignment_ = val; }
        private:
            std::map<unsigned int,std::string> access_names_;
            unsigned int id_;
            std::string name_;
            std::string scalartype_;
            unsigned int scalartype_size_;
            unsigned int alignment_;
        };







        class kernel_argument;

        class infos_base{
        public:
            typedef std::string repr_t;
            virtual std::string generate(unsigned int i) const { return ""; }
            virtual repr_t repr() const = 0;
            virtual repr_t simplified_repr() const = 0;
            virtual ~infos_base(){ }
        };


        class op_infos_base : public infos_base{
        public:
            bool is_assignment() const { return is_assignment_; }
            repr_t repr() const{ return name_;}
            repr_t simplified_repr() const { return name_; }
        protected:
            op_infos_base(std::string const & name, bool is_assignment) : name_(name), is_assignment_(is_assignment){ }
        private:
            std::string name_;
            bool is_assignment_;
        };

        class nonarithmetic_op_infos_base : public op_infos_base{
        public:
            nonarithmetic_op_infos_base(std::string name) : op_infos_base(name,false){ }
        };

        class arithmetic_op_infos_base : public op_infos_base{
        public:
            std::string generate(unsigned int i) const{ return expr_; }
        protected:
            arithmetic_op_infos_base( std::string const & expr, std::string const & name, bool is_assignment) :  op_infos_base(name,is_assignment), expr_(expr){ }
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

            op_infos_base* op_reduce(){ return op_reduce_.get(); }

            std::string generate(unsigned int i) const{ return op_reduce_->generate(i); }

        private:
            viennacl::tools::shared_ptr<op_infos_base> op_reduce_;
        };

        class trans_type : public nonarithmetic_op_infos_base{
        public:
            trans_type() : nonarithmetic_op_infos_base("trans"){ }
        };

        class unary_tree_infos_base{
        public:
            infos_base & sub(){ return *sub_; }

            std::string generate(unsigned int i) const{
                return "(-" + sub_->generate(i) +")";
            }

        protected:
            unary_tree_infos_base(infos_base * sub) : sub_(sub){        }
            viennacl::tools::shared_ptr<infos_base> sub_;
        };



        class binary_tree_infos_base : public virtual infos_base{
        public:
            infos_base & lhs() const{ return *lhs_; }
            infos_base & rhs() const{ return *rhs_; }
            op_infos_base & op() { return *op_; }
            infos_base::repr_t repr() const { return "p_"+lhs_->repr() + op_->repr() + rhs_->repr()+"_p"; }
            infos_base::repr_t simplified_repr() const { return "p_"+lhs_->simplified_repr() + op_->repr() + rhs_->simplified_repr()+"_p"; }

        protected:
            binary_tree_infos_base(infos_base * lhs, op_infos_base * op, infos_base * rhs) : lhs_(lhs), op_(op), rhs_(rhs){        }
            viennacl::tools::shared_ptr<infos_base> lhs_;
            viennacl::tools::shared_ptr<op_infos_base> op_;
            viennacl::tools::shared_ptr<infos_base> rhs_;
        };


        class arithmetic_tree_infos_base : public binary_tree_infos_base{
        public:
            std::string generate(unsigned int i) const { return "(" + lhs_->generate(i) + op_->generate(i) + rhs_->generate(i) + ")"; }
            infos_base::repr_t simplified_repr() const{
                if(op_->is_assignment()){
                    return "p_"+lhs_->simplified_repr() + assign_type().repr() + rhs_->simplified_repr()+"_p";
                }
                else{
                    return lhs_->repr();
                }
            }
            arithmetic_tree_infos_base( infos_base * lhs, op_infos_base* op, infos_base * rhs) :  binary_tree_infos_base(lhs,op,rhs){        }
        private:
        };

        class vector_expression_infos_base : public arithmetic_tree_infos_base{
        public:
            vector_expression_infos_base( infos_base * lhs, op_infos_base* op, infos_base * rhs) : arithmetic_tree_infos_base( lhs,op,rhs){ }
        };

        class scalar_expression_infos_base : public arithmetic_tree_infos_base{
        public:
            scalar_expression_infos_base( infos_base * lhs, op_infos_base* op, infos_base * rhs) : arithmetic_tree_infos_base( lhs,op,rhs){ }
        };

        class matrix_expression_infos_base : public arithmetic_tree_infos_base{
        public:
            matrix_expression_infos_base( infos_base * lhs, op_infos_base* op, infos_base * rhs) : arithmetic_tree_infos_base( lhs,op,rhs){ }
        };


        class kernel_argument : public virtual infos_base{
        public:
            virtual viennacl::backend::mem_handle const & handle() const = 0;
            kernel_argument( ) { }
            void access_name(unsigned int i, std::string const & new_name) { infos_->access_name(i,new_name); }
            virtual ~kernel_argument(){ }
            virtual std::string generate(unsigned int i) const { return infos_->access_name(i); }
            std::string name() const { return infos_->name(); }
            std::string const & scalartype() const { return infos_->scalartype(); }
            unsigned int scalartype_size() const { return infos_->scalartype_size(); }
            infos_base::repr_t simplified_repr() const { return repr(); }
            std::string aligned_scalartype() const {
                unsigned int alignment = infos_->alignment();
                std::string const & scalartype = infos_->scalartype();
                if(alignment==1){
                    return scalartype;
                }
                else{
                    assert( (alignment==2 || alignment==4 || alignment==8 || alignment==16) && "Invalid alignment");
                    return scalartype + to_string(alignment);
                }
            }
            unsigned int alignment() const { return infos_->alignment(); }
            void alignment(unsigned int val) { infos_->alignment(val); }
            virtual std::string arguments_string() const = 0;
            virtual void enqueue(unsigned int & arg, viennacl::ocl::kernel & k) const = 0;
        protected:
            shared_infos_t* infos_;
        };


        class cpu_scal_infos_base : public kernel_argument{
        public:
            virtual std::string arguments_string() const{
                return scalartype() + " " + name();
            }
        };

        class gpu_scal_infos_base : public kernel_argument{
        public:
            virtual std::string arguments_string() const{
                return  "__global " + scalartype() + "*"  + " " + name();
            }
        };

        class matmat_prod_infos_base : public matrix_expression_infos_base{
        public:
            matmat_prod_infos_base( infos_base * lhs, op_infos_base* op, infos_base * rhs, std::string const & f_expr) :
                matrix_expression_infos_base(lhs,op,rhs),f_expr_(f_expr){
                val_name_ = repr() + "_val";
            }

            repr_t simplified_repr() const { return binary_tree_infos_base::simplified_repr(); }

            std::string val_name(unsigned int m, unsigned int n){
                return val_name_ +  '_' + to_string(m) + '_' + to_string(n);
            }

            std::string update_val(std::string const & res, std::string const & lhs, std::string const & rhs){
                std::string expr(f_expr_);
                replace_all_occurences(expr,"#1",lhs);
                replace_all_occurences(expr,"#2",rhs);
                return res + " = " + res + op_->generate(0) + "(" + expr + ")";

            }

            std::string make_expr(std::string const & lhs, std::string const & rhs){
                std::string res(f_expr_);
                replace_all_occurences(res,"#1",lhs);
                replace_all_occurences(res,"#2",rhs);
                return res;
            }

//            op_infos_base const & op_reduce(){ return *op_reduce_; }


        private:
            std::string f_expr_;
            std::string val_name_;
        };

        class inprod_infos_base : public scalar_expression_infos_base, public kernel_argument{
        public:
            enum step_t{compute,reduce};
            step_t step(){ return *step_; }
            void step(step_t s){ *step_ = s; }
            local_memory<1> make_local_memory(unsigned int size){
                return local_memory<1>(name()+"_local",size,scalartype());
            }
            repr_t repr() const{ return "aa"; }

            infos_base::repr_t simplified_repr() const { return scalar_expression_infos_base::simplified_repr(); }

            std::string arguments_string() const{
                return "__global " + scalartype() + "*" + " " + name();
            }
            std::string generate(unsigned int i) const{
                if(*step_==compute){
                    return sum_name() + " += " "dot((" + lhs_->generate(i) +  ")" " , " "(" + rhs_->generate(i) + "))" ;
                }
                return infos_->access_name(0);
            }
            std::string sum_name() const{
                return name()+"_sum";
            }

            unsigned int n_groupsize_used_for_compute(){ return 0; }

        protected:
            inprod_infos_base(infos_base * lhs
                              , infos_base * rhs
                              , std::string const & f_expr
                              ,step_t * step): scalar_expression_infos_base(lhs,new inner_prod_type(),rhs), f_expr_(f_expr), step_(step){

            }


        private:
            std::string f_expr_;
            viennacl::tools::shared_ptr<step_t> step_;
        };


        class vec_infos_base : public kernel_argument{
        public:
            std::string  size() const{ return name() + "_size"; }
//            std::string  internal_size() const{ return name() + "_internal_size";}
            std::string  start() const{ return name() + "_start";}
            std::string  inc() const{ return name() + "_inc";}
            std::string arguments_string() const{ return  " __global " + aligned_scalartype() + "*"  + " " + name()
                                                                     + ", unsigned int " + size();                                                                     ;
                                                }
            virtual size_t real_size() const = 0;
            virtual ~vec_infos_base(){ }
        protected:
            vec_infos_base() : kernel_argument() { }
        };


        class mat_infos_base : public kernel_argument{
        public:
            std::string  internal_size1() const{ return name() +"internal_size1_"; }
            std::string  internal_size2() const{ return name() +"internal_size2_"; }
            std::string  row_inc() const{ return name() +"row_inc_"; }
            std::string  col_inc() const{ return name() +"col_inc_";}
            std::string  row_start() const{ return name() +"row_start_";}
            std::string  col_start() const{ return name() +"col_start_";}
            std::string arguments_string() const{
                return " __global " + aligned_scalartype() + "*"  + " " + name()
                                                            + ", unsigned int " + row_start()
                                                            + ", unsigned int " + col_start()
                                                            + ", unsigned int " + row_inc()
                                                            + ", unsigned int " + col_inc()
                                                            + ", unsigned int " + internal_size1()
                                                            + ", unsigned int " + internal_size2();
            }
            bool const is_rowmajor() const { return is_rowmajor_; }
            bool const is_transposed() const { return is_transposed_; }
            std::string offset(std::string const & offset_i, std::string const & offset_j){
                if(is_rowmajor_){
                    return '(' + offset_i + ')' + '*' + internal_size2() + "+ (" + offset_j + ')';
                }
                return '(' + offset_i + ')' + "+ (" + offset_j + ')' + '*' + internal_size1();
            }

            virtual size_t real_size1() const = 0;
            virtual size_t real_size2() const = 0;
            virtual ~mat_infos_base() { }
        protected:
            mat_infos_base(bool is_rowmajor
                           ,bool is_transposed) : kernel_argument()
                                                  ,is_rowmajor_(is_rowmajor)
                                                  ,is_transposed_(is_transposed){ }
        protected:
            bool is_rowmajor_;
            bool is_transposed_;
        };

        class function_base : public infos_base{
        protected:
            typedef std::map<std::string,viennacl::tools::shared_ptr<infos_base> > args_map_t;
        public:
            function_base(std::string const & name) : name_(name){ }
            virtual std::string name() const {
                return name_;
            }

            repr_t repr() const{
                repr_t res;
                for(args_map_t::const_iterator it = args_map_.begin() ; it != args_map_.end() ; ++it){
                    res += it->second->repr();

                }
                return res;
            }

            repr_t simplified_repr() const{
                    repr_t res;
                    for(args_map_t::const_iterator it = args_map_.begin() ; it != args_map_.end() ; ++it){
                        res += it->second->simplified_repr();

                    }
                    return res;
            }

            std::list<infos_base*> args() const{
                std::list<infos_base*> res;
                for(args_map_t::const_iterator it = args_map_.begin() ; it!= args_map_.end() ; ++it)
                    res.push_back(it->second.get());
                return res;
            }

        protected:
            std::string name_;
            args_map_t args_map_;
        };

        template<class T1, class T2=void, class T3=void>
        class symbolic_function : public function_base{
        public:
            typedef typename T1::ScalarType ScalarType;

            symbolic_function(std::string const & name,std::string const & expr) : function_base(name), expr_(expr){
            }


            template<class T>
            void add_arg(std::string const & arg_name, T const & t){
                args_map_.insert(std::make_pair(arg_name, new T(t)));
            }


            virtual std::string generate(unsigned int i) const {
                std::string res(expr_);
                for(args_map_t::const_iterator it = args_map_.begin() ; it!= args_map_.end() ; ++it)
                    replace_all_occurences(res,it->first,it->second->generate(i));
                return res;
            }


        private:
            std::string expr_;
        };


        static bool operator<(infos_base const & first, infos_base const & other){
            if(binary_tree_infos_base const * t = dynamic_cast<binary_tree_infos_base const *>(&first)){
                return t->lhs() < other || t->rhs() < other;
            }
            else if(binary_tree_infos_base const * p= dynamic_cast<binary_tree_infos_base const *>(&other)){
                return first < p->lhs() || first < p->rhs();
            }
            else if(kernel_argument const * t = dynamic_cast<kernel_argument const *>(&first)){
                  if(kernel_argument const * p = dynamic_cast<kernel_argument const*>(&other)){
                     return t->handle() < p->handle();
                  }
            }
            return false;
       }


        template<class T, class Pred>
        static void extract_as(infos_base* root, std::set<T*, deref_less> & args, Pred pred){
            if(arithmetic_tree_infos_base* p = dynamic_cast<arithmetic_tree_infos_base*>(root)){
                extract_as(&p->lhs(), args,pred);
                extract_as(&p->rhs(),args,pred);
            }
            else if(function_base* p = dynamic_cast<function_base*>(root)){
                std::list<infos_base*> func_args(p->args());
                for(std::list<infos_base*>::const_iterator it = func_args.begin(); it!= func_args.end(); ++it){
                    extract_as(*it,args,pred);
                }
            }
            else if(inprod_infos_base* p = dynamic_cast<inprod_infos_base*>(root)){
                if(p->step() == inprod_infos_base::compute){
                    extract_as(&p->lhs(), args,pred);
                    extract_as(&p->rhs(),args,pred);
                }
            }
            if(T* t = dynamic_cast<T*>(root))
                if(pred(t)) args.insert(t);
        }

        template<class T>
        static unsigned int count_type(infos_base* root){
            unsigned int res = 0;
            if(arithmetic_tree_infos_base* p = dynamic_cast<arithmetic_tree_infos_base*>(root)){
                res += count_type<T>(&p->lhs());
                res += count_type<T>(&p->rhs());
            }
            else if(function_base* p = dynamic_cast<function_base*>(root)){
                std::list<infos_base*> func_args(p->args());
                for(std::list<infos_base*>::const_iterator it = func_args.begin(); it!= func_args.end(); ++it){
                    res += count_type<T>(*it);
                }
            }
            else if(inprod_infos_base* p = dynamic_cast<inprod_infos_base*>(root)){
                if(p->step() == inprod_infos_base::compute){
                    res += count_type<T>(&p->lhs());
                    res += count_type<T>(&p->rhs());
                }
            }
            if(T* t = dynamic_cast<T*>(root)) return res+1;
            else return res;
        }

    }

}
#endif // SYMBOLIC_TYPES_BASE_HPP
