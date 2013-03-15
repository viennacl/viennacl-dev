#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP


#include "viennacl/forwards.h"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/backend/mem_handle.hpp"

#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/forwards.h"
#include "viennacl/generator/operators.hpp"

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

            std::string access(std::string const & i, std::string const & j) const{
                return name() + "[" + '('+i+')' + '*' + to_string(size2_) + "+ (" + j + ") ]";
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
            virtual std::string generate(unsigned int i) const { return ""; }
            virtual std::string repr() const = 0;
            virtual std::string simplified_repr() const = 0;
            virtual ~infos_base(){ }
        };






        class binary_tree_infos_base : public virtual infos_base{
        public:
            infos_base & lhs() const{ return *lhs_; }
            infos_base & rhs() const{ return *rhs_; }
            binary_op_infos_base & op() { return *op_; }
            std::string repr() const { return "p_"+lhs_->repr() + op_->name() + rhs_->repr()+"_p"; }
            std::string simplified_repr() const { return "p_"+lhs_->simplified_repr() + op_->name() + rhs_->simplified_repr()+"_p"; }

        protected:
            binary_tree_infos_base(infos_base * lhs, binary_op_infos_base * op, infos_base * rhs) : lhs_(lhs), op_(op), rhs_(rhs){        }
            viennacl::tools::shared_ptr<infos_base> lhs_;
            viennacl::tools::shared_ptr<binary_op_infos_base> op_;
            viennacl::tools::shared_ptr<infos_base> rhs_;
        };


        class binary_arithmetic_tree_infos_base : public binary_tree_infos_base{
        public:
            std::string generate(unsigned int i) const { return "(" +  op_->generate(lhs_->generate(i), rhs_->generate(i) ) + ")"; }
            std::string simplified_repr() const{
                if(op_->is_assignment()){
                    return "p_"+lhs_->simplified_repr() + assign_type().name() + rhs_->simplified_repr()+"_p";
                }
                else{
                    return lhs_->repr();
                }
            }
            binary_arithmetic_tree_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) :  binary_tree_infos_base(lhs,op,rhs){        }
        private:
        };

        class binary_vector_expression_infos_base : public binary_arithmetic_tree_infos_base{
        public:
            binary_vector_expression_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) : binary_arithmetic_tree_infos_base( lhs,op,rhs){ }
        };

        class binary_scalar_expression_infos_base : public binary_arithmetic_tree_infos_base{
        public:
            binary_scalar_expression_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) : binary_arithmetic_tree_infos_base( lhs,op,rhs){ }
        };

        class binary_matrix_expression_infos_base : public binary_arithmetic_tree_infos_base{
        public:
            binary_matrix_expression_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) : binary_arithmetic_tree_infos_base( lhs,op,rhs){ }
        };


        class unary_tree_infos_base : public virtual infos_base{
        public:
            unary_tree_infos_base(infos_base * sub, unary_op_infos_base * op) : sub_(sub), op_(op) { }
            infos_base & sub() const{ return *sub_; }
            unary_op_infos_base & op() { return *op_; }
            std::string repr() const { return "p_"+ op_->name() + sub_->repr()+"_p"; }
            std::string simplified_repr() const { return "p_" + op_->name() + sub_->simplified_repr()+"_p"; }
            std::string generate(unsigned int i) const { return "(" +  op_->generate(sub_->generate(i)) + ")"; }
        protected:
            viennacl::tools::shared_ptr<infos_base> sub_;
            viennacl::tools::shared_ptr<unary_op_infos_base> op_;
        };

        class unary_vector_expression_infos_base : public unary_tree_infos_base{
        public:
            unary_vector_expression_infos_base( infos_base * sub, unary_op_infos_base* op) : unary_tree_infos_base( sub,op){ }
        };

        class unary_scalar_expression_infos_base : public unary_tree_infos_base{
        public:
            unary_scalar_expression_infos_base( infos_base * sub, unary_op_infos_base* op) : unary_tree_infos_base( sub,op){ }
        };

        class unary_matrix_expression_infos_base : public unary_tree_infos_base{
        public:
            unary_matrix_expression_infos_base( infos_base * sub, unary_op_infos_base* op) : unary_tree_infos_base( sub,op){ }
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
            std::string simplified_repr() const { return repr(); }
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

        class matmat_prod_infos_base : public binary_matrix_expression_infos_base{
        public:
            matmat_prod_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs, std::string const & f_expr) :
                binary_matrix_expression_infos_base(lhs,op,rhs),f_expr_(f_expr){
                val_name_ = repr() + "_val";
            }

            std::string simplified_repr() const { return binary_tree_infos_base::simplified_repr(); }

            std::string val_name(unsigned int m, unsigned int n){
                return val_name_ +  '_' + to_string(m) + '_' + to_string(n);
            }

            std::string update_val(std::string const & res, std::string const & lhs, std::string const & rhs){
                std::string expr(f_expr_);
                replace_all_occurences(expr,"#1",lhs);
                replace_all_occurences(expr,"#2",rhs);
                return res + " = " + op_->generate(res ,"(" + expr + ")");

            }

            std::string make_expr(std::string const & lhs, std::string const & rhs){
                std::string res(f_expr_);
                replace_all_occurences(res,"#1",lhs);
                replace_all_occurences(res,"#2",rhs);
                return res;
            }
        private:
            std::string f_expr_;
            std::string val_name_;
        };


        class matvec_prod_infos_base : public binary_vector_expression_infos_base{
        public:
            matvec_prod_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs, std::string const & f_expr) :
                binary_vector_expression_infos_base(lhs,op,rhs),f_expr_(f_expr){
                val_name_ = repr() + "_val";
            }

            std::string simplified_repr() const { return binary_tree_infos_base::simplified_repr(); }

            std::string val_name(unsigned int m, unsigned int n){
                return val_name_ +  '_' + to_string(m) + '_' + to_string(n);
            }

            std::string update_val(std::string const & res, std::string const & lhs, std::string const & rhs){
                std::string expr(f_expr_);
                replace_all_occurences(expr,"#1",lhs);
                replace_all_occurences(expr,"#2",rhs);
                return res + " = " + op_->generate(res ,"("+expr+")");

            }

            std::string make_expr(std::string const & lhs, std::string const & rhs){
                std::string res(f_expr_);
                replace_all_occurences(res,"#1",lhs);
                replace_all_occurences(res,"#2",rhs);
                return res;
            }
        private:
            std::string f_expr_;
            std::string val_name_;
        };

        class inprod_infos_base : public binary_scalar_expression_infos_base, public kernel_argument{
        public:
            enum step_t{compute,reduce};

            inprod_infos_base(infos_base * lhs
                              , binary_op_infos_base * op
                              , infos_base * rhs
                              , std::string const & f_expr
                              ,step_t * step): binary_scalar_expression_infos_base(lhs,op,rhs), f_expr_(f_expr), step_(step){

            }

            step_t step(){ return *step_; }

            void step(step_t s){ *step_ = s; }

            local_memory<1> make_local_memory(unsigned int size){
                return local_memory<1>(name()+"_local",size,scalartype());
            }
            std::string repr() const{
                return binary_scalar_expression_infos_base::repr();
            }

            std::string simplified_repr() const {
                return binary_scalar_expression_infos_base::simplified_repr();
            }

            std::string arguments_string() const{
                return "__global " + scalartype() + "*" + " " + name();
            }

            std::string update_val(unsigned int i){
                std::string expr(f_expr_);
                replace_all_occurences(expr,"#1",lhs_->generate(i));
                replace_all_occurences(expr,"#2",rhs_->generate(i));
                return sum_name() + " = " + op_->generate(sum_name(), "(" + expr + ")");

            }

            std::string generate(unsigned int i) const{
                return infos_->access_name(0);
            }
            std::string sum_name() const{
                return name()+"_sum";
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

            std::string repr() const{
                std::string res;
                for(args_map_t::const_iterator it = args_map_.begin() ; it != args_map_.end() ; ++it){
                    res += it->second->repr();

                }
                return res;
            }

            std::string simplified_repr() const{
                    std::string res;
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
            if(binary_arithmetic_tree_infos_base* p = dynamic_cast<binary_arithmetic_tree_infos_base*>(root)){
                extract_as(&p->lhs(), args,pred);
                extract_as(&p->rhs(),args,pred);
            }
            else if(unary_tree_infos_base* p = dynamic_cast<unary_tree_infos_base*>(root)){
                extract_as(&p->sub(), args,pred);
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
            if(binary_arithmetic_tree_infos_base* p = dynamic_cast<binary_arithmetic_tree_infos_base*>(root)){
                res += count_type<T>(&p->lhs());
                res += count_type<T>(&p->rhs());
            }
            else if(unary_tree_infos_base* p = dynamic_cast<unary_tree_infos_base*>(root)){
                res += count_type<T>(&p->sub());
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
            if(dynamic_cast<T*>(root)) return res+1;
            else return res;
        }

    }

}
#endif // SYMBOLIC_TYPES_BASE_HPP
