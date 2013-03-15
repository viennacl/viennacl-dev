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
#include "viennacl/generator/functions.hpp"


namespace viennacl
{
  namespace generator
  {


      template<class T> struct repr_of;
      template<> struct repr_of<float>{ static const std::string value(){ return "f"; } };
      template<> struct repr_of<double>{ static const std::string value(){ return "d"; } };


      template<class LHS, class RHS, class OP_REDUCE>
      class matmat_prod_infos : public matmat_prod_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          matmat_prod_infos(LHS const & lhs, RHS const & rhs, std::string const & f_expr) : matmat_prod_infos_base(new LHS(lhs), new prod_type<OP_REDUCE>(), new RHS(rhs), f_expr){ }
      private:

      };

      template<class LHS, class RHS, class OP_REDUCE>
      class matvec_prod_infos : public matvec_prod_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          matvec_prod_infos(LHS const & lhs, RHS const & rhs, std::string const & f_expr) : matvec_prod_infos_base(new LHS(lhs), new prod_type<OP_REDUCE>(), new RHS(rhs), f_expr){ }
      private:

      };

      template<class LHS, class RHS, class OP_REDUCE>
      class inprod_infos : public inprod_infos_base{
          typedef typename LHS::ScalarType ScalarType;
      public:
          template<class SharedInfosMapT, class TemporariesMapT>
          inprod_infos(SharedInfosMapT & shared_infos,
                       TemporariesMapT & temporaries_map,
                       LHS const & lhs, RHS const & rhs,
                       std::string const & f_expr):
              inprod_infos_base(new LHS(lhs), new prod_type<OP_REDUCE>, new RHS(rhs), f_expr
                                ,new step_t(inprod_infos_base::compute)), tmp_(1024){
              temporaries_map.insert(std::make_pair(this,tmp_.handle())).first;
              infos_= &shared_infos.insert(std::make_pair(tmp_.handle(),shared_infos_t(shared_infos.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))).first->second;
          }

           void enqueue(unsigned int & arg, viennacl::ocl::kernel & k) const{
               k.arg(arg++,tmp_.handle().opencl_handle());
           }

           viennacl::backend::mem_handle const & handle() const{ return tmp_.handle(); }

      private:
          viennacl::vector<ScalarType> tmp_;
      };

      template<class LHS, class OP, class RHS>
      class binary_vector_expression : public binary_vector_expression_infos_base{
      public:
          typedef typename LHS::ScalarType ScalarType;
          binary_vector_expression(LHS const & lhs, RHS const & rhs) :binary_vector_expression_infos_base( new LHS(lhs),new OP(),new RHS(rhs)){ }
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
        cpu_symbolic_scalar(ScalarType val) : cpu_scal_infos_base(print_type<SCALARTYPE>::value()), val_(val){ }
    private:
        ScalarType val_;
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

        template<class SharedInfosMapT>
        gpu_symbolic_scalar(SharedInfosMapT & map, vcl_t const & vcl_scal) : vcl_scal_(vcl_scal){
            infos_= &map.insert(std::make_pair(vcl_scal_.handle(),shared_infos_t(map.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))).first->second;
        }

        viennacl::backend::mem_handle const & handle() const{ return vcl_scal_.handle(); }

        void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const{
            k.arg(n_arg++,vcl_scal_);
        }

        std::string repr() const{
            return "gs"+repr_of<SCALARTYPE>::value();
        }


    private:
        vcl_t const & vcl_scal_;
    };


    /**
      * @brief Symbolic vector type
      *
      * @tparam SCALARTYPE The Scalartype of the vector in the generated code
      * @tparam ALIGNMENT The Alignment of the vector in the generated code
      */
      template <typename SCALARTYPE>
      class symbolic_vector : public vec_infos_base{
        private:
          typedef symbolic_vector<SCALARTYPE> self_type;
        public:
          typedef viennacl::vector<SCALARTYPE> vcl_vec_t;
          typedef SCALARTYPE ScalarType;

          template<class SharedInfosMapT>
          symbolic_vector(SharedInfosMapT & map
                          ,vcl_vec_t const & vcl_vec) : vcl_vec_(vcl_vec){
              infos_= &map.insert(std::make_pair(vcl_vec_.handle(),shared_infos_t(map.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))).first->second;
          }
          virtual viennacl::backend::mem_handle const & handle() const{ return vcl_vec_.handle(); }

          void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const{
              k.arg(n_arg++,vcl_vec_);
              k.arg(n_arg++,cl_uint(vcl_vec_.internal_size()/infos_->alignment()));
          }

          std::string repr() const{
              return "v"+repr_of<SCALARTYPE>::value();
          }

          size_t real_size() const{ return vcl_vec_.size(); }

        private:
          vcl_vec_t const & vcl_vec_;
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
          typedef typename VCL_MATRIX::value_type::value_type ScalarType;

          template<class SharedInfosMapT>
          symbolic_matrix(SharedInfosMapT & map
                          ,VCL_MATRIX const & vcl_mat
                          ,bool is_transposed) : mat_infos_base(are_same_type<typename VCL_MATRIX::orientation_category,viennacl::row_major_tag>::value
                                                                       ,is_transposed), vcl_mat_(vcl_mat){
              infos_= &map.insert(std::make_pair(vcl_mat_.handle(),shared_infos_t(map.size(),print_type<ScalarType>::value(),sizeof(ScalarType)))).first->second;

          }

          size_t real_size1() const{
              return vcl_mat_.size1();
          }

          size_t real_size2() const{
              return vcl_mat_.size2();
          }

          void enqueue(unsigned int & n_arg, viennacl::ocl::kernel & k) const{
              cl_uint size1_arg = cl_uint(vcl_mat_.internal_size1());
              cl_uint size2_arg = cl_uint(vcl_mat_.internal_size2());

              if(is_rowmajor_) size2_arg /= infos_->alignment();
              else size1_arg /= infos_->alignment();

              k.arg(n_arg++,vcl_mat_);
              k.arg(n_arg++,cl_uint(0));
              k.arg(n_arg++,cl_uint(1));
              k.arg(n_arg++,cl_uint(0));
              k.arg(n_arg++,cl_uint(1));
              k.arg(n_arg++,size1_arg);
              k.arg(n_arg++,size2_arg);
          }


          viennacl::backend::mem_handle const & handle() const{ return vcl_mat_.handle(); }

          std::string repr() const{
              return "m"+repr_of<typename VCL_MATRIX::value_type::value_type>::value()+'_'+to_string((int)is_rowmajor_)+'_'+to_string((int)is_transposed_);
          }
      private:
          VCL_MATRIX const & vcl_mat_;

      };





  } // namespace generator
} // namespace viennacl


#endif
