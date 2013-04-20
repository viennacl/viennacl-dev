#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_BASE_HPP


#include "viennacl/ocl/utils.hpp"

#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/operators.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/backend/memory.hpp"
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
            std::string declare() const{ return "__local " + scalartype_ + " " + name_ + '[' + to_string(size_) + ']'; }
            unsigned int size() const{ return size_; }
            std::string const & name() const{ return name_; }
            std::string access(std::string const & index) const{ return name_ + '[' + index + ']'; }
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
            std::string declare() const{ return impl_.declare(); }
            std::string const & name() const { return impl_.name(); }
            unsigned int size1() const{ return size1_; }
            unsigned int size2() const{ return size2_; }
            std::string access(std::string const & i, std::string const & j) const{ return name() + "[" + '('+i+')' + '*' + to_string(size2_) + "+ (" + j + ") ]";}
        private:
            unsigned int size1_;
            unsigned int size2_;
            local_memory<1> impl_;
        };

        struct shared_infos_t{
        public:
            shared_infos_t(unsigned int _id, std::string const & _scalartype, unsigned int _scalartype_size, unsigned int _alignment = 1) {
                id = _id;
                name = "arg" + to_string(id);
                scalartype = _scalartype;
                scalartype_size = _scalartype_size;
                alignment = _alignment;
            }
            std::map<unsigned int,std::string> access_index;
            std::map<unsigned int,std::string> private_values;
            unsigned int id;
            std::string name;
            std::string scalartype;
            unsigned int scalartype_size;
            unsigned int alignment;
        };

        class kernel_argument;
        class symbolic_datastructure;

        class infos_base{
        public:
            virtual std::string generate(unsigned int i, int vector_element = -1) const { return ""; }
            virtual std::string repr() const = 0;
            virtual std::string simplified_repr() const = 0;
            virtual void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > >  & shared_infos, code_generation::optimization_profile* prof)= 0;
            virtual void access_index(unsigned int i, std::string const & ind0, std::string const & ind1) = 0;
            virtual void fetch(unsigned int i, kernel_generation_stream & kss) = 0;
            virtual void write_back(unsigned int i, kernel_generation_stream & kss) = 0;
            virtual void get_kernel_arguments(std::vector<kernel_argument const *> & args) const = 0;
            virtual bool operator==(infos_base const & other) const = 0;
            virtual ~infos_base(){ }
            infos_base() : current_kernel_(0) { }
        protected:
            unsigned int current_kernel_;
        };



        class binary_tree_infos_base : public virtual infos_base{
        public:
            infos_base & lhs() const{ return *lhs_; }
            infos_base & rhs() const{ return *rhs_; }
            binary_op_infos_base & op() { return *op_; }
            std::string repr() const { return op_->name() + "("+lhs_->repr() + "," + rhs_->repr() +")"; }
            std::string simplified_repr() const {
                if(assignment_op_infos_base* opa = dynamic_cast<assignment_op_infos_base*>(opa))
                    return "assign(" + lhs_->repr() + "," + rhs_->repr() + ")";
                else
                    return lhs_->repr();
            }
            void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > >  & shared_infos, code_generation::optimization_profile* prof){
                lhs_->bind(shared_infos,prof);
                rhs_->bind(shared_infos,prof);
            }
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
                lhs_->access_index(i,ind0,ind1);
                rhs_->access_index(i,ind0,ind1);
            }
            void fetch(unsigned int i, kernel_generation_stream & kss){
                lhs_->fetch(i,kss);
                rhs_->fetch(i,kss);
            }
            void get_kernel_arguments(std::vector<kernel_argument const *> & args) const{
                lhs_->get_kernel_arguments(args);
                rhs_->get_kernel_arguments(args);
            }

            void write_back(unsigned int i, kernel_generation_stream & kss){
                if(dynamic_cast<assignment_op_infos_base*>(op_.get())) lhs_->write_back(i,kss);
            }

            bool operator==(infos_base const & other) const{
                if(binary_tree_infos_base const * p = dynamic_cast<binary_tree_infos_base const *>(&other)){
                    return *lhs_==*p->lhs_ && op_->name()==p->op_->name() && *rhs_==*p->rhs_;
                }
                return false;
            }
        protected:
            binary_tree_infos_base(infos_base * lhs, binary_op_infos_base * op, infos_base * rhs) : lhs_(lhs), op_(op), rhs_(rhs){        }
            viennacl::tools::shared_ptr<infos_base> lhs_;
            viennacl::tools::shared_ptr<binary_op_infos_base> op_;
            viennacl::tools::shared_ptr<infos_base> rhs_;
        };


        class binary_arithmetic_tree_infos_base : public binary_tree_infos_base{
        public:
            std::string generate(unsigned int i, int vector_element = -1) const {
                std::map<unsigned int, std::string>::const_iterator it = override_generation_.find(i);
                if(it==override_generation_.end())
                    return "(" +  op_->generate(lhs_->generate(i,vector_element), rhs_->generate(i,vector_element) ) + ")";
                else
                    return it->second;
            }
            binary_arithmetic_tree_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) :  binary_tree_infos_base(lhs,op,rhs){        }
            void override_generation(unsigned int i, std::string const & new_val){ override_generation_[i] = new_val; }
        private:
            std::map<unsigned int, std::string> override_generation_;
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
            unary_op_infos_base const & op() const{ return *op_; }
            std::string repr() const { return op_->name() + "("+ sub_->repr()+")"; }
            std::string simplified_repr() const { return repr(); }
            void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > >  & shared_infos, code_generation::optimization_profile* prof){
                sub_->bind(shared_infos,prof);
            }
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
                if(dynamic_cast<trans_type *>(op_.get())) sub_->access_index(i,ind1,ind0);
                else  sub_->access_index(i,ind0,ind1);
            }
            void fetch(unsigned int i, kernel_generation_stream & kss){ sub_->fetch(i,kss); }
            void write_back(unsigned int i, kernel_generation_stream & kss){  }
            void get_kernel_arguments(std::vector<kernel_argument const *> & args) const{
                sub_->get_kernel_arguments(args);
            }
            std::string generate(unsigned int i, int vector_element = -1) const { return "(" +  op_->generate(sub_->generate(i,vector_element)) + ")"; }
            bool operator==(infos_base const & other) const{
                if(unary_tree_infos_base const * p = dynamic_cast<unary_tree_infos_base const *>(&other)){
                    return *sub_==*p->sub_ && op_->name()==p->op_->name();
                }
                return false;
            }
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


        class kernel_argument{
        protected:
            virtual void const * handle() const = 0;
        public:
            kernel_argument(std::string const & address_space, std::string const & scalartype_name, std::string const & name) : address_space_(address_space), scalartype_name_(scalartype_name), name_(name){ }
            virtual void enqueue(unsigned int & arg, viennacl::ocl::kernel & k) const = 0;
            bool operator==(kernel_argument const & other) const{ return name_ == other.name_; }
            virtual std::string repr() const = 0;
            std::string const & name() const { return name_; }
            void scalartype_name(std::string const & str) { scalartype_name_ = str; }
        protected:
            std::string address_space_;
            std::string scalartype_name_;
            std::string name_;
        };

        class value_argument_base: public kernel_argument{
        public:
            value_argument_base(std::string const & scalartype_name, std::string const & name) : kernel_argument("",scalartype_name,name){ }
            std::string repr() const{ return address_space_ + " " + scalartype_name_ + " " + name_; }
        };

        template<class ScalarType>
        class value_argument : public value_argument_base{
        public:
            typedef typename viennacl::result_of::cl_type<ScalarType>::type cl_type;
        private:
            void const * handle() const { return static_cast<void const *>(&handle_); }
        public:
            value_argument(std::string const & name, ScalarType const & val) : value_argument_base(print_type<ScalarType>::value(), name), handle_(val){ }
            void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const { k.arg(n_arg++,handle_); }
        private:
            cl_type handle_;
        };


        class pointer_argument_base : public kernel_argument{
        public:
            pointer_argument_base(std::string const & address_space, std::string const & scalartype_name, std::string const & name) : kernel_argument(address_space,scalartype_name,name){ }
            std::string repr() const{ return address_space_ + " " + scalartype_name_ + "* " + name_; }
        };

        template<class ScalarType>
        class pointer_argument : public pointer_argument_base{
        private:
            void const * handle() const { return static_cast<void const *>(&handle_); }
        public:
            pointer_argument(std::string const & name, viennacl::backend::mem_handle const & handle, unsigned int alignment=1) : pointer_argument_base("__global", print_type<ScalarType>::value() + ((alignment>1)?to_string(alignment):""), name), handle_(handle){ }
            void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const { k.arg(n_arg++,handle_.opencl_handle()); }
        private:
            viennacl::backend::mem_handle const & handle_;
        };

        class symbolic_datastructure : public virtual infos_base{
        public:
            void private_value(unsigned int i, std::string const & new_name) { infos_->private_values[i] = new_name; }
            void clear_private_value(unsigned int i) { infos_->private_values[i] = ""; }
            std::string name() const { return infos_->name; }
            std::string const & scalartype() const { return infos_->scalartype; }
            unsigned int scalartype_size() const { return infos_->scalartype_size; }
            std::string simplified_repr() const { return repr(); }
            std::string aligned_scalartype() const {
                unsigned int alignment = infos_->alignment;
                std::string const & scalartype = infos_->scalartype;
                if(alignment==1){
                    return scalartype;
                }
                else{
                    assert( (alignment==2 || alignment==4 || alignment==8 || alignment==16) && "Invalid alignment");
                    return scalartype + to_string(alignment);
                }
            }
            unsigned int alignment() const { return infos_->alignment; }
            void alignment(unsigned int val) { infos_->alignment = val; }
            void get_kernel_arguments(std::vector<kernel_argument const *>& args) const{
                for(std::vector<tools::shared_ptr<kernel_argument> >::const_iterator it = other_arguments_.begin() ; it != other_arguments_.end() ; ++it){
                    unique_push_back(args,(kernel_argument const *)it->get());
                }
            }
            virtual ~symbolic_datastructure(){ }
        protected:
            shared_infos_t* infos_;
            std::vector<tools::shared_ptr<kernel_argument> > other_arguments_;
        };

        class buffered_datastructure : public symbolic_datastructure{
        protected:
            virtual std::string access_buffer(unsigned int i) const = 0;
        public:
            std::string get_access_index(unsigned int i) const { return infos_->access_index[i]; }
            void fetch(unsigned int i, kernel_generation_stream & kss){
                if(infos_->private_values[i].empty()){
                    std::string val = infos_->name + "_private";
                    std::string aligned_scalartype = infos_->scalartype;
                    if(infos_->alignment > 1) aligned_scalartype += to_string(infos_->alignment);
                    kss << aligned_scalartype << " " << val << " = " << access_buffer(i) << ";" << std::endl;
                    infos_->private_values[i] = val;
                }
            }
            virtual void write_back(unsigned int i, kernel_generation_stream & kss){
                kss << access_buffer(i) << " = " << infos_->private_values[i] << ";" << std::endl;
                infos_->private_values[i].clear();
            }
            std::string generate(unsigned int i, int vector_element = -1) const {
                std::string res;
                if(infos_->private_values[i].empty()) res = access_buffer(i);
                else res = infos_->private_values[i];
                if(vector_element >= 0 && infos_->alignment > 1) res += ".s" + to_string(vector_element);
                return res;
            }
        };

        class cpu_scal_infos_base : public symbolic_datastructure{
        public:
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
            std::string generate(unsigned int i, int vector_element = -1) const { return infos_->name; }
            void fetch(unsigned int i, kernel_generation_stream & kss){ }
            void write_back(unsigned int i, kernel_generation_stream & kss){ }
        };

        class gpu_scal_infos_base : public buffered_datastructure{
        public:
            std::string generate(unsigned int i, int vector_element = -1) const {
                if(infos_->private_values[i].empty()) return "*"+infos_->name;
                else  return infos_->private_values[i];
            }
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
        };

        class vec_infos_base : public buffered_datastructure{
        public:
            vec_infos_base(size_t size) : size_(size) { }
            std::string  size() const{ return name() + "_size"; }
            size_t real_size() const { return size_; }
            virtual ~vec_infos_base(){ }
        protected:
            size_t size_;
        };


        class mat_infos_base : public buffered_datastructure{
        public:
            mat_infos_base(bool is_rowmajor) : is_rowmajor_(is_rowmajor){ }
            virtual size_t real_size1() const = 0;
            virtual size_t real_size2() const = 0;
            std::string  internal_size1() const{ return name() +"internal_size1_"; }
            std::string  internal_size2() const{ return name() +"internal_size2_"; }
            bool const is_rowmajor() const { return is_rowmajor_; }
            std::string offset(std::string const & offset_i, std::string const & offset_j){
                if(is_rowmajor_){
                    return '(' + offset_i + ')' + '*' + internal_size2() + "+ (" + offset_j + ')';
                }
                return '(' + offset_i + ')' + "+ (" + offset_j + ')' + '*' + internal_size1();
            }
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
                std::string str;
                if(is_rowmajor_)
                    str = ind0+"*"+internal_size2()+"+"+ind1;
                else
                    str = ind1+"*"+internal_size1()+"+"+ind0;
                infos_->access_index[i] = str;
            }
        protected:
            bool is_rowmajor_;
        };

        class matmat_prod_infos_base : public binary_matrix_expression_infos_base{
        public:
            matmat_prod_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) :
                binary_matrix_expression_infos_base(lhs,op,rhs){
            }
            void set_val_name(std::string const & val_name) { val_name_ = val_name; }
            std::string simplified_repr() const { return binary_tree_infos_base::simplified_repr(); }
            std::string val_name(unsigned int m, unsigned int n){ return val_name_ +  '_' + to_string(m) + '_' + to_string(n); }
            std::string update_val(std::string const & res, std::string const & lhs, std::string const & rhs){ return res + " = " + op_->generate(res , lhs + "*" + rhs); }
        private:
            std::string val_name_;
        };


        class matvec_prod_infos_base : public binary_vector_expression_infos_base{
        public:
            matvec_prod_infos_base( infos_base * lhs, binary_op_infos_base* op, infos_base * rhs) : binary_vector_expression_infos_base(lhs,new mul_type,rhs), op_reduce_(op){            }
            std::string simplified_repr() const { return binary_tree_infos_base::simplified_repr(); }
            binary_op_infos_base const & op_reduce() const { return *op_reduce_; }
        private:
            viennacl::tools::shared_ptr<binary_op_infos_base> op_reduce_;
        };

        class inner_product_infos_base : public binary_scalar_expression_infos_base, public buffered_datastructure{
        public:
            inner_product_infos_base(infos_base * lhs, binary_op_infos_base * op, infos_base * rhs): binary_scalar_expression_infos_base(lhs,new mul_type,rhs)
                                                                                                    , op_reduce_(op){ }
            bool is_computed(){ return current_kernel_; }
            void set_computed(){ current_kernel_ = 1; }
            std::string repr() const{ return binary_scalar_expression_infos_base::repr(); }
            void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > >  & shared_infos, code_generation::optimization_profile* prof){ binary_scalar_expression_infos_base::bind(shared_infos,prof); }
            std::string simplified_repr() const { return binary_scalar_expression_infos_base::simplified_repr(); }
            void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){  binary_scalar_expression_infos_base::access_index(i,ind0,ind1); }
            void fetch(unsigned int i, kernel_generation_stream & kss){ binary_scalar_expression_infos_base::fetch(i,kss); }
            virtual void write_back(unsigned int i, kernel_generation_stream & kss){ binary_scalar_expression_infos_base::write_back(i,kss); }
            binary_op_infos_base const & op_reduce() const { return *op_reduce_; }
            void get_kernel_arguments(std::vector<kernel_argument const *> & args) const{
                buffered_datastructure::get_kernel_arguments(args);
                if(current_kernel_==0) binary_scalar_expression_infos_base::get_kernel_arguments(args);
            }
            std::string generate(unsigned int i, int vector_element = -1) const{ return binary_scalar_expression_infos_base::generate(i,vector_element); }
        private:
            viennacl::tools::shared_ptr<binary_op_infos_base> op_reduce_;
        };

//        template<class T, class Pred>
//        static void extract_as(infos_base* root, std::set<T*, deref_less> & args, Pred pred){
//            if(binary_arithmetic_tree_infos_base* p = dynamic_cast<binary_arithmetic_tree_infos_base*>(root)){
//                extract_as(&p->lhs(), args,pred);
//                extract_as(&p->rhs(),args,pred);
//            }
//            else if(unary_tree_infos_base* p = dynamic_cast<unary_tree_infos_base*>(root)){
//                extract_as(&p->sub(), args,pred);
//            }
//            if(T* t = dynamic_cast<T*>(root))
//                if(pred(t)) args.insert(t);
//        }

        template<class T, class Pred>
        static void extract_as(infos_base* root, std::list<T*> & args, Pred pred){
            if(binary_arithmetic_tree_infos_base* p = dynamic_cast<binary_arithmetic_tree_infos_base*>(root)){
                extract_as(&p->lhs(), args,pred);
                extract_as(&p->rhs(),args,pred);
            }
            else if(unary_tree_infos_base* p = dynamic_cast<unary_tree_infos_base*>(root)){
                extract_as(&p->sub(), args,pred);
            }
            if(T* t = dynamic_cast<T*>(root)){
                if(pred(t)) args.push_back(t);
            }
        }

        template<class T, class Pred>
        static void extract_as_unique(infos_base* root, std::list<T*> & args, Pred pred){
            if(binary_arithmetic_tree_infos_base* p = dynamic_cast<binary_arithmetic_tree_infos_base*>(root)){
                extract_as(&p->lhs(), args,pred);
                extract_as(&p->rhs(),args,pred);
            }
            else if(unary_tree_infos_base* p = dynamic_cast<unary_tree_infos_base*>(root)){
                extract_as(&p->sub(), args,pred);
            }
            if(T* t = dynamic_cast<T*>(root)){
                if(pred(t)) unique_push_back(args,t);
            }
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
            if(dynamic_cast<T*>(root)) return res+1;
            else return res;
        }

        template<class T>
        bool is_transposed(T const * t){
            if(unary_matrix_expression_infos_base const * m = dynamic_cast<unary_matrix_expression_infos_base const *>(t)){
                return static_cast<bool>(dynamic_cast<trans_type const *>(&m->op()));
            }
            if(unary_vector_expression_infos_base const * v = dynamic_cast<unary_vector_expression_infos_base const *>(t))
                return static_cast<bool>(dynamic_cast<trans_type const *>(&v->op()));
            return false;
        }
    }

}
#endif // SYMBOLIC_TYPES_BASE_HPP
