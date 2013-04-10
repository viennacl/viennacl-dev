#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */



#include "viennacl/generator/symbolic_types_base.hpp"
#include "viennacl/generator/operators.hpp"
#include "viennacl/generator/code_generation/optimization_profile.hpp"

namespace viennacl
{
  namespace generator
  {


          template<class T> struct repr_of;
          template<> struct repr_of<float>{ static const std::string value(){ return "f"; } };
          template<> struct repr_of<double>{ static const std::string value(){ return "d"; } };
          template<> struct repr_of<int>{ static const std::string value(){ return "i"; } };
          template<> struct repr_of<unsigned int>{ static const std::string value(){ return "ui"; } };


      template<class T, class Enable=void>
      struct to_sym{
          typedef T type;
          static type result(T const & t){ return t; }
      };

      template<class T>
      static typename to_sym<T>::type make_sym(T const & t){
          return to_sym<T>::result(t);
      }

      template<class LHS, class OP, class RHS>
      class binary_vector_expression : public binary_vector_expression_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          binary_vector_expression(LHS const & lhs, RHS const & rhs) :binary_vector_expression_infos_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
      };



      template<class LHS, class RHS, class OP_REDUCE>
      class binary_vector_expression<LHS,reduce_type<OP_REDUCE>,RHS> : public matvec_prod_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          binary_vector_expression(LHS const & lhs, RHS const & rhs) : matvec_prod_infos_base(new LHS(lhs), new reduce_type<OP_REDUCE>(), new RHS(rhs)){ }
      };


      template<class LHS, class OP, class RHS>
      class binary_scalar_expression : public binary_scalar_expression_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          typedef LHS Lhs;
          typedef RHS Rhs;
          binary_scalar_expression(LHS const & lhs, RHS const & rhs) :binary_scalar_expression_infos_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
      };



      template<class LHS, class OP, class RHS>
      class binary_matrix_expression : public binary_matrix_expression_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          typedef LHS Lhs;
          typedef RHS Rhs;
          binary_matrix_expression(LHS const & lhs, RHS const & rhs) :binary_matrix_expression_infos_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
      };

      template<class LHS, class RHS, class OP_REDUCE>
      class binary_matrix_expression<LHS,reduce_type<OP_REDUCE>,RHS> : public matmat_prod_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          binary_matrix_expression(LHS const & lhs, RHS const & rhs) : matmat_prod_infos_base(new LHS(lhs), new reduce_type<OP_REDUCE>(), new RHS(rhs)){ }
      private:

      };

      template<class SUB, class OP>
      class unary_vector_expression : public unary_vector_expression_infos_base{
      public:
          typedef typename SUB::ScalarType ScalarType;
          unary_vector_expression(SUB const & sub) :unary_vector_expression_infos_base(new SUB(sub), new OP()){ }
      };

      template<class SUB, class OP>
      class unary_scalar_expression : public unary_scalar_expression_infos_base{
      public:
          typedef typename SUB::ScalarType ScalarType;
          unary_scalar_expression(SUB const & sub) :unary_scalar_expression_infos_base(new SUB(sub), new OP()){ }
      };

      template<class LHS, class OP_REDUCE, class RHS>
      class binary_scalar_expression<LHS, reduce_type<OP_REDUCE>, RHS > : public inner_product_infos_base{
          typedef typename LHS::ScalarType ScalarType;
      public:
           binary_scalar_expression(LHS const & lhs, RHS const & rhs):  inner_product_infos_base(new LHS(lhs), new OP_REDUCE, new RHS(rhs)), tmp_(1024){ }
           void enqueue(unsigned int & arg, viennacl::ocl::kernel & k) const{ k.arg(arg++,tmp_.handle().opencl_handle()); }
           void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
               infos_= unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this
                                                                  ,tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))))->second.get();
               other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new pointer_argument<ScalarType>(infos_->name, tmp_.handle())));
               lhs_->bind(shared_infos, prof);
               rhs_->bind(shared_infos, prof);
           }
           bool operator==(symbolic_datastructure const & other) const{
               if(binary_scalar_expression const * p = dynamic_cast<binary_scalar_expression const *>(&other)){
                   return tmp_.handle().opencl_handle() == p->tmp_.handle().opencl_handle();
               }
               return false;
           }
           void const * handle() const{ return static_cast<void const *>(&tmp_.handle()); }
      private:
          viennacl::vector<ScalarType> tmp_;
      };

      template<class SUB, class OP>
      class unary_matrix_expression : public unary_matrix_expression_infos_base{
      public:
          typedef typename SUB::ScalarType ScalarType;
          unary_matrix_expression(SUB const & sub) :unary_matrix_expression_infos_base(new SUB(sub), new OP()){ }
      };






    /**
    * @brief Symbolic scalar type. Will be passed by value.
    *
    * @tparam SCALARTYPE The Scalartype of the scalar in the generated code
    */
    template <typename SCALARTYPE>
    class cpu_symbolic_scalar : public cpu_scal_infos_base
    {
    public:
        typedef SCALARTYPE ScalarType;
        cpu_symbolic_scalar(ScalarType const & val) :  val_(val){}
        std::string repr() const{ return "vscal"+repr_of<SCALARTYPE>::value(); }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos
                  ,code_generation::optimization_profile * prof){
            infos_= unique_insert(shared_infos,std::make_pair((symbolic_datastructure *)this,
                                                              tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))))->second.get();
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<ScalarType>(name(), val_)));
        }
        bool operator==(symbolic_datastructure const & other) const{
            if(cpu_symbolic_scalar const * p = dynamic_cast<cpu_symbolic_scalar const *>(&other)){
                return val_ == p->val_;
            }
            return false;
        }
        void const * handle() const { return static_cast<void const *>(&val_); }
        std::string generate(unsigned int i, int vector_element = -1) const{  return infos_->name; }
    private:
        ScalarType val_;
    };

    template<int N>
    class symbolic_constant : public infos_base{
    public:
        virtual std::string generate(unsigned int i, int vector_element = -1) const { return to_string(N); }
        virtual std::string repr() const { return "cst"+to_string(N); }
        virtual std::string simplified_repr() const { return repr(); }
        virtual void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){ }
        void fetch(unsigned int i, kernel_generation_stream & kss){ }
        void write_back(unsigned int i, kernel_generation_stream & kss){ }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > >& , code_generation::optimization_profile*){ }
        bool operator==(symbolic_datastructure const & other) const{ return dynamic_cast<symbolic_constant *>(&other); }
        void get_kernel_arguments(std::vector<kernel_argument const * > & args) const { }
    };


    /**
    * @brief Symbolic scalar type. Will be passed by pointer.
    *
    * @tparam ID The argument ID of the scalar in the generated code
    * @tparam SCALARTYPE The SCALARTYPE of the scalar in the generated code
    */
    template <typename SCALARTYPE>
    class gpu_symbolic_scalar : public gpu_scal_infos_base
    {
      private:
        typedef gpu_symbolic_scalar<SCALARTYPE> self_type;

      public:
        typedef viennacl::scalar<SCALARTYPE> vcl_t;
        typedef SCALARTYPE ScalarType;
        gpu_symbolic_scalar(vcl_t const & vcl_scal) : vcl_scal_(vcl_scal){ }
        void const * handle() const{ return static_cast<void const *>(&vcl_scal_.handle()); }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
            infos_= unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this,
                                                               tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))))->second.get();
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new pointer_argument<ScalarType>(name(), vcl_scal_.handle())));
        }
        bool operator==(symbolic_datastructure const & other) const{
            if(gpu_symbolic_scalar const * p = dynamic_cast<gpu_symbolic_scalar const *>(&other)){
                return vcl_scal_.handle().opencl_handle() == p->vcl_scal_.handle().opencl_handle();
            }
            return false;
        }
        std::string repr() const{ return "pscal"+repr_of<SCALARTYPE>::value(); }
    private:
        vcl_t const & vcl_scal_;
    };







      /**
      * @brief Symbolic matrix type
      *
      * @tparam SCALARTYPE The Scalartype of the matrix in the generated code
      * @tparam F The Layout of the matrix in the generated code
      * @tparam ALIGNMENT The Alignment of the matrix in the generated code
      */
      template<class VCL_MATRIX>
      class symbolic_matrix : public mat_infos_base
      {
          typedef symbolic_matrix<VCL_MATRIX> self_type;
        public:
          typedef VCL_MATRIX vcl_t;
          typedef typename vcl_t::value_type::value_type ScalarType;
          symbolic_matrix(VCL_MATRIX const & vcl_mat) : mat_infos_base(are_same_type<typename VCL_MATRIX::orientation_category,viennacl::row_major_tag>::value), vcl_mat_(vcl_mat){ }
          size_t real_size1() const{ return vcl_mat_.internal_size1(); }
          size_t real_size2() const{ return vcl_mat_.internal_size2(); }
          void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
              infos_= unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this,
                                                                 tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType), prof->vectorization()))))->second.get();
              other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new pointer_argument<ScalarType>(name(), vcl_mat_.handle(), infos_->alignment)));
              cl_uint size1_arg = cl_uint(real_size1());
              cl_uint size2_arg = cl_uint(real_size2());
              if(is_rowmajor_) size2_arg /= infos_->alignment;
              else size1_arg /= infos_->alignment;
              other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(internal_size1(), size1_arg)));
              other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(internal_size2(), size2_arg)));
          }
          std::string repr() const{ return "mat"+repr_of<typename VCL_MATRIX::value_type::value_type>::value()+(is_rowmajor_?'R':'C'); }
          bool operator==(symbolic_datastructure const & other) const{
              if(symbolic_matrix const * p = dynamic_cast<symbolic_matrix const *>(&other))
                  return typeid(other)==typeid(*this) &&vcl_mat_.handle().opencl_handle() == p->vcl_mat_.handle().opencl_handle();
              return false;
          }
      protected:
          VCL_MATRIX const & vcl_mat_;
      };


    template<class VCL_MATRIX>
    class replicate_matrix : public mat_infos_base{
    private:
        std::string access_buffer(unsigned int i) const { return name()+"_underlying" + '[' + infos_->access_index[i] + ']'; }
    public:
        typedef VCL_MATRIX vcl_t;
        typedef typename vcl_t::value_type::value_type ScalarType;
        replicate_matrix(VCL_MATRIX const & sub,unsigned int m,unsigned int k) : mat_infos_base(are_same_type<typename VCL_MATRIX::orientation_category,viennacl::row_major_tag>::value)
                                                                                ,underlying_(sub)
                                                                                , m_(m) , k_(k){ }
        void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
            std::string new_ind0 = ind0;
            std::string new_ind1 = ind1;
            std::string underlying_size1 = name() + "_underlying_size1";
            std::string underlying_size2 = name() + "_underlying_size2";
            if(m_>1) new_ind0 += "%" + underlying_size1;
            if(k_>1) new_ind1 += "%" + underlying_size2;
            std::string str;
            if(is_rowmajor_)
                str = new_ind0+"*"+underlying_size2+"+"+new_ind1;
            else
                str = new_ind1+"*"+underlying_size1+"+"+new_ind0;
            infos_->access_index[i] = str;
        }

        std::string repr() const{  return "repmat"+repr_of<typename VCL_MATRIX::value_type::value_type>::value()+(is_rowmajor_?'R':'C'); }
        void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
            infos_= unique_insert(shared_infos,std::make_pair( (symbolic_datastructure *)this,
                                                               tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType), prof->vectorization()))))->second.get();

            {
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new pointer_argument<ScalarType>(name()+"_underlying", underlying_.handle(), infos_->alignment)));
            cl_uint size1_arg = cl_uint(underlying_.internal_size1());
            cl_uint size2_arg = cl_uint(underlying_.internal_size2());
            if(is_rowmajor_) size2_arg /= infos_->alignment;
            else size1_arg /= infos_->alignment;
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(name()+"_underlying_size1", size1_arg)));
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(name()+"_underlying_size2", size2_arg)));
            }

            {
            cl_uint size1_arg = cl_uint(real_size1());
            cl_uint size2_arg = cl_uint(real_size2());
            if(is_rowmajor_) size2_arg /= infos_->alignment;
            else size1_arg /= infos_->alignment;
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(internal_size1(), size1_arg)));
            other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(internal_size2(), size2_arg)));
            }

        }
        size_t real_size1() const{ return m_*underlying_.internal_size1(); }
        size_t real_size2() const{ return k_*underlying_.internal_size2(); }
        bool operator==(symbolic_datastructure const & other) const{
            if(replicate_matrix const * p = dynamic_cast<replicate_matrix const *>(&other))
                return typeid(other)==typeid(*this)
                        && m_==p->m_
                        && k_==p->k_
                        && underlying_.handle()==p->underlying_.handle();
            return false;
        }
    private:
        VCL_MATRIX const & underlying_;
        unsigned int m_;
        unsigned int k_;
    };
      
     /**
       * @brief Symbolic vector type
       *
       * @tparam SCALARTYPE The Scalartype of the vector in the generated code
       * @tparam ALIGNMENT The Alignment of the vector in the generated code
       */
       template <typename SCALARTYPE>
       class symbolic_vector : public vec_infos_base{
       public:
           typedef SCALARTYPE ScalarType;
       protected:
           virtual void add_additional_arguments(){ }
       public:
           typedef viennacl::vector<SCALARTYPE> vcl_t;
           symbolic_vector(viennacl::vector<SCALARTYPE> const & vcl_vec) : vcl_vec_(vcl_vec){ }
           virtual void const * handle() const{ return static_cast<void const *>(&vcl_vec_.handle()); }
           void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
               infos_= unique_insert(shared_infos,std::make_pair((symbolic_datastructure *)this,
                                                                 tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType), prof->vectorization()))))->second.get();
               other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new pointer_argument<ScalarType>(name(), vcl_vec_.handle(), infos_->alignment)));
               other_arguments_.push_back(tools::shared_ptr<kernel_argument>(new value_argument<unsigned int>(size(), vcl_vec_.internal_size()/infos_->alignment)));
               add_additional_arguments();
           }
           std::string repr() const{ return "vec"+repr_of<SCALARTYPE>::value(); }
           size_t real_size() const{ return vcl_vec_.size(); }
           bool operator==(symbolic_datastructure const & other) const{
               if(symbolic_vector const * p = dynamic_cast<symbolic_vector const *>(&other))
                   return typeid(other)==typeid(*this)
                           &&vcl_vec_.handle().opencl_handle() == p->vcl_vec_.handle().opencl_handle();
               return false;
           }

         protected:
           viennacl::vector<SCALARTYPE> const & vcl_vec_;
      };

      template<class SCALARTYPE>
      class symbolic_shift : public symbolic_vector<SCALARTYPE>{
        public:
          typedef typename symbolic_vector<SCALARTYPE>::ScalarType ScalarType;
        private:
          virtual void add_additional_arguments(){
              symbolic_vector<ScalarType>::other_arguments_.push_back(tools::shared_ptr<kernel_argument>(
                                                                          new value_argument<int>(symbolic_vector<ScalarType>::infos_->name+"_shift",val_)
                                                                          ));
          }
        public:
          symbolic_shift(viennacl::vector<ScalarType> const & vcl_vec, int val) : symbolic_vector<SCALARTYPE>(vcl_vec), val_(val){ }
          std::string repr() const{ return "shift"+repr_of<ScalarType>::value(); }
          void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
              std::string x = "(int)"+ ind0 + " + " + symbolic_vector<SCALARTYPE>::infos_->name + "_shift";
              std::string min = "0";
              std::string max = "(int)"+symbolic_vector<ScalarType>::size() + "-1";
              std::string ind = "min(max("+x+","+min+"),"+max+")";
              symbolic_vector<ScalarType>::infos_->access_index[i] = ind;
          }
          bool operator==(symbolic_datastructure const & other) const{
              if(symbolic_shift const * p = dynamic_cast<symbolic_shift const *>(&other))
                  return typeid(other)==typeid(*this)
                          &&symbolic_vector<SCALARTYPE>::vcl_vec_.handle().opencl_handle() == p->vcl_vec_.handle().opencl_handle()
                          &&val_ == p->val_;
              return false;
          }

        private:
          int val_;
      };

      template<class SCALARTYPE>
      class symbolic_range : public symbolic_vector<SCALARTYPE>{
        public:
          typedef typename symbolic_vector<SCALARTYPE>::ScalarType ScalarType;
        private:
          virtual void add_additional_arguments(){
              argoffset_ = new value_argument<int>(symbolic_vector<ScalarType>::infos_->name+"_offset",offset_);
              symbolic_vector<ScalarType>::other_arguments_.push_back(tools::shared_ptr<kernel_argument>(argoffset_));
          }
        public:
          symbolic_range(viennacl::vector<ScalarType> const & vcl_vec, int offset) : symbolic_vector<SCALARTYPE>(vcl_vec), offset_(offset){ }
          std::string repr() const{ return "rangevec"+repr_of<ScalarType>::value(); }
          void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
              symbolic_vector<ScalarType>::infos_->access_index[i] = ind0 + " + " + argoffset_->name();
          }
        private:
          int offset_;
          value_argument<int> * argoffset_;
      };

      template<class VCL_MATRIX>
      class symbolic_diag : public vec_infos_base{
      public:
          typedef typename VCL_MATRIX::value_type::value_type ScalarType;
      public:
          symbolic_diag(VCL_MATRIX const & vcl_mat) : vcl_mat_(vcl_mat){ }
          virtual void const * handle() const{ return static_cast<void const *>(this); }
          void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const{
              k.arg(n_arg++,vcl_mat_);
              k.arg(n_arg++,cl_uint(vcl_mat_.internal_size1()));
          }
          void bind(std::vector< std::pair<symbolic_datastructure *, tools::shared_ptr<shared_infos_t> > > & shared_infos, code_generation::optimization_profile* prof){
              infos_= &unique_insert(shared_infos,std::make_pair((symbolic_datastructure *)this,
                                                                  tools::shared_ptr<shared_infos_t>(new shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))))->second;
          }
          std::string repr() const{
              bool is_rowmajor = are_same_type<typename VCL_MATRIX::orientation_category,viennacl::row_major_tag>::value;
              return "diag"+repr_of<ScalarType>::value() + (is_rowmajor?'R':'C');
          }
          size_t real_size() const{ return vcl_mat_.size1(); }
          void access_index(unsigned int i, std::string const & ind0, std::string const & ind1){
              infos_->access_index[i] = ind0+"*"+size()+"+"+ind0;
          }
          template<typename RHS_TYPE>
          binary_vector_expression<symbolic_diag<VCL_MATRIX>, inplace_add_type, typename to_sym<RHS_TYPE>::type >
          operator+= ( RHS_TYPE const & rhs ){
            return binary_vector_expression<symbolic_diag<VCL_MATRIX>, inplace_add_type, typename to_sym<RHS_TYPE>::type >(*this,make_sym(rhs));
          }
        protected:
          VCL_MATRIX const & vcl_mat_;
      };

      



  } // namespace generator
} // namespace viennacl


#endif
