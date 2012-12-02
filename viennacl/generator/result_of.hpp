#ifndef VIENNACL_GENERATOR_TRAITS_RESULT_OF_HPP
#define VIENNACL_GENERATOR_TRAITS_RESULT_OF_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file viennacl/generator/result_of.hpp
 *  @brief Provides a set of metafunctions for type deductions within the kernel generator framework. Experimental.
 *
 *  Generator code contributed by Philippe Tillet
 */

#include <string>

#include "viennacl/generator/operators.hpp"
#include "viennacl/generator/forwards.h"
#include "viennacl/generator/meta_tools/utils.hpp"
#include "viennacl/generator/elementwise_modifier.hpp"
#include "viennacl/ocl/local_mem.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/forwards.h"
#include "CL/cl.h"

namespace viennacl
{
  namespace generator
  {
    namespace result_of 
    {

      class runtime_wrapper
      {
        protected:
          std::string name_;
          int arg_id_;
          
        public:
            runtime_wrapper(std::string const & _name, int _arg_id)
             : name_(_name), arg_id_(_arg_id) {}
             
            virtual ~runtime_wrapper() {}
           
          int arg_id() const { return arg_id_; }
          std::string name() const { return name_; }
          
          virtual void enqueue(unsigned int arg_pos, 
                               viennacl::ocl::kernel & k,
                               std::map<unsigned int, viennacl::any> & runtime_args,
                               std::map<std::string, viennacl::ocl::handle<cl_mem> > & temporaries) = 0;
      };

      class shared_memory_wrapper : public runtime_wrapper
      {
        public:
            shared_memory_wrapper() : runtime_wrapper( "shared_memory_ptr", -1 ){ }
  
          void enqueue(unsigned int arg_pos,
                       viennacl::ocl::kernel & k,
                       std::map<unsigned int, viennacl::any> & /* runtime_args */,
                       std::map<std::string, viennacl::ocl::handle<cl_mem> > & /* temporaries */)
          {
            unsigned int lmem_size = k.local_work_size();
            #ifdef VIENNACL_DEBUG_CUSTOM_OPERATION
            std::cout << "Enqueuing Local memory of size " << lmem_size << " at pos " << arg_pos << std::endl;
            #endif
            k.arg(arg_pos, viennacl::ocl::local_mem(lmem_size*sizeof(float)));
          }
  
      };

      template <class T, class SIZE_T>
      struct vector_runtime_wrapper : public runtime_wrapper 
      {
        private:
          unsigned int size_id_;
          
          template<typename ScalarType, unsigned int Alignment>
          typename SIZE_T::size_type size(viennacl::vector<ScalarType,Alignment> * size_arg) { return size_arg->size(); }

          template<typename ScalarType, class F, unsigned int Alignment>
          typename SIZE_T::size_type size(viennacl::matrix<ScalarType,F,Alignment> * size_arg) { return size_arg->size2(); }
          
          template<typename ScalarType, unsigned int Alignment>
          typename SIZE_T::size_type internal_size(viennacl::vector<ScalarType,Alignment> * size_arg) { return size_arg->internal_size(); }

          template<typename ScalarType, class F, unsigned int Alignment>
          typename SIZE_T::size_type internal_size(viennacl::matrix<ScalarType,F,Alignment> * size_arg) { return size_arg->internal_size2(); }
          
        public:
          vector_runtime_wrapper(std::string const & _name, int _arg_id, unsigned int _size_id)
            : runtime_wrapper(_name,_arg_id),size_id_(_size_id) {}
            
          void enqueue(unsigned int arg_pos,
                       viennacl::ocl::kernel & k,
                       std::map<unsigned int, viennacl::any> & runtime_args,
                       std::map<std::string, 
                       viennacl::ocl::handle<cl_mem> > & /*temporaries*/)
          { 
            SIZE_T * size_arg = viennacl::any_cast<SIZE_T * >(runtime_args[size_id_]);
            viennacl::ocl::handle<cl_mem> handle = NULL;
            T * current_arg = viennacl::any_cast<T * >(runtime_args[arg_id_]);
            handle = current_arg->handle().opencl_handle();

            k.arg(arg_pos, handle );
            k.arg(arg_pos+1,cl_uint(size(size_arg)));
            k.arg(arg_pos+2,cl_uint(internal_size(size_arg)));
          }
      };

      template <class T, class SIZE_DESCRIPTOR>
      struct vector_expression : public runtime_wrapper
      {
        typedef T type;
        typedef typename SIZE_DESCRIPTOR::ScalarType ScalarType;
        static const unsigned int Alignment = SIZE_DESCRIPTOR::Alignment;

        static runtime_wrapper * runtime_descriptor()
        {
          return new vector_runtime_wrapper<viennacl::vector<ScalarType,Alignment>,
                                            typename SIZE_DESCRIPTOR::runtime_type>(T::name(),
                                                                                    T::id,SIZE_DESCRIPTOR::id);
        }
        
        static std::string size_expression() 
        {
          return SIZE_DESCRIPTOR::size2_name();
        }
        
        static std::string internal_size_expression() 
        {
          return SIZE_DESCRIPTOR::internal_size2_name() + "/" + to_string(Alignment);
        }

        static int n_args()
        {
          return 3;
        }
      };

      template <class T, class SIZE1_T, class SIZE2_T>
      struct matrix_runtime_wrapper : public runtime_wrapper
      {
        private:
          unsigned int size1_id_;
          unsigned int size2_id_;
        public:
          matrix_runtime_wrapper(std::string const & _name,
                                 int _arg_id,
                                 unsigned int _size1_id,
                                 unsigned int _size2_id) 
                                : runtime_wrapper(_name,_arg_id), size1_id_(_size1_id), size2_id_(_size2_id) {}
                                
          unsigned int n_elements(){ return size1_id_*size2_id_; }
          
          void enqueue(unsigned int arg_pos,
                       viennacl::ocl::kernel & k,
                       std::map<unsigned int, viennacl::any> & runtime_args,
                       std::map<std::string,
                       viennacl::ocl::handle<cl_mem> > & /*temporaries*/)
          { 
            T * current_arg = any_cast<T * >(runtime_args[arg_id_]);
            SIZE1_T * size1_arg = any_cast<SIZE1_T * >(runtime_args[size1_id_]);
            SIZE2_T * size2_arg = any_cast<SIZE2_T * >(runtime_args[size2_id_]);
            k.arg(arg_pos, current_arg->handle().opencl_handle());
            k.arg(arg_pos+1,cl_uint(0));
            k.arg(arg_pos+2,cl_uint(0));
            k.arg(arg_pos+3,cl_uint(size1_arg->size1()));
            k.arg(arg_pos+4,cl_uint(size2_arg->size2()));
            k.arg(arg_pos+5,cl_uint(size1_arg->size1()));
            k.arg(arg_pos+6,cl_uint(size2_arg->size2()));
            k.arg(arg_pos+7,cl_uint(size1_arg->internal_size1()));
            k.arg(arg_pos+8,cl_uint(size2_arg->internal_size2()));
          }
      };
          
      template <class T, class SIZE1_DESCRIPTOR, class SIZE2_DESCRIPTOR>
      struct matrix_expression 
      {
        typedef T type;
        typedef typename SIZE1_DESCRIPTOR::ScalarType ScalarType;
        typedef typename SIZE1_DESCRIPTOR::Layout Layout;
        static const unsigned int Alignment = SIZE1_DESCRIPTOR::Alignment;
        
        static runtime_wrapper * runtime_descriptor()
        {
          return new matrix_runtime_wrapper<viennacl::matrix<ScalarType,Layout,Alignment>,
                                            typename SIZE1_DESCRIPTOR::runtime_type,
                                            typename SIZE2_DESCRIPTOR::runtime_type>(T::name(),T::id,SIZE1_DESCRIPTOR::id,
                                                                                     SIZE2_DESCRIPTOR::id);
        }
        
        static std::string size_expression()
        {
          return size1_expression() + "*" + size2_expression();
        }

        static std::string size1_expression() 
        {
          return SIZE1_DESCRIPTOR::size1_name();
        }

        static std::string size2_expression() 
        {
          return SIZE2_DESCRIPTOR::size2_name();
        }

        static std::string internal_size1_expression() 
        {
          return SIZE1_DESCRIPTOR::internal_size1_name() + "/" + to_string(Alignment);
        }

        static std::string internal_size2_expression() 
        {
          return SIZE2_DESCRIPTOR::internal_size2_name() + "/" + to_string(Alignment);
        }

        static int n_args()
        {
          return 9;
        }
      };

//      template <class T>
//      struct scalar_size_descriptor
//      {
//        static unsigned int size(viennacl::ocl::kernel & k) { return 1; }
//      };

//      template <class LHS, class RHS>
//      struct scalar_size_descriptor<compound_node<LHS,inner_prod_type,RHS> >
//      {
//        static unsigned int size(viennacl::ocl::kernel & k)
//        {
//          return k.global_work_size(0)/k.local_work_size(0);
//        }
//      };

      template <class T>
      struct scalar_runtime_wrapper: public runtime_wrapper
      {
      private:
        bool is_inner_product_;
      public:
        typedef typename T::ScalarType ScalarType;
        
        scalar_runtime_wrapper( std::string const & _name, int _arg_id, bool is_inner_product) : runtime_wrapper(_name,_arg_id),is_inner_product_(is_inner_product){}
        
        void enqueue(unsigned int arg_pos,
                     viennacl::ocl::kernel & k,
                     std::map<unsigned int,
                     viennacl::any> & runtime_args, 
                     std::map<std::string, 
                     viennacl::ocl::handle<cl_mem> > & temporaries)
        {
          if(is_inner_product_)
          {
            if(temporaries.find(name_)==temporaries.end())
            {
              temporaries.insert(
                std::make_pair(name_,
                          viennacl::ocl::handle<cl_mem>(
                          viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE,
                                                                        k.global_work_size(0)/k.local_work_size(0)*sizeof(ScalarType))
                                                      )
                        )
                      );
            }
            k.arg(arg_pos, temporaries[name_]);
          }
      
          if(arg_id_==-2)
              k.arg(arg_pos, temporaries[name_]);
          else
          {
            viennacl::scalar<ScalarType>* current_arg = any_cast<viennacl::scalar<ScalarType> * >(runtime_args[arg_id_]);
            k.arg(arg_pos, current_arg->handle().opencl_handle());
          }

        }
      };

      template <unsigned int ID, class ScalarType>
      struct scalar_runtime_wrapper<viennacl::generator::cpu_symbolic_scalar<ID, ScalarType> >: public runtime_wrapper
      {
        scalar_runtime_wrapper(std::string const & _name, int _arg_id, bool /*is_inner_product*/) : runtime_wrapper(_name,_arg_id){ }
        
        void enqueue(unsigned int arg_pos,
                     viennacl::ocl::kernel & k,
                     std::map<unsigned int, viennacl::any> & runtime_args,
                     std::map<std::string, viennacl::ocl::handle<cl_mem> > & /*temporaries*/)
        {
          ScalarType* current_arg = any_cast<ScalarType * >(runtime_args[arg_id_]);
          k.arg(arg_pos, static_cast<typename viennacl::tools::cl_type<ScalarType>::Result>(*current_arg));
        }
      };
          
      template <class T>
      struct scalar_expression 
      {
        typedef T type;
        typedef typename T::ScalarType ScalarType;
        static const unsigned int Alignment=1;

        static runtime_wrapper * runtime_descriptor()
        {
          return new scalar_runtime_wrapper<T>(T::name(),T::id,result_of::is_inner_product_leaf<T>::value || result_of::is_inner_product_impl<T>::value);
        }

        static int n_args()
        {
          return 1;
        }
      };

      template<class T>
      struct constant_expression
      {
        static const long value = T::value;
      };

      template <class T>
      struct expression_type
      {
        typedef NullType Result;
      };



      /*
       * Compound Nodes - General case
       */

      template <class LHS, class OP, class RHS>
      struct expression_type<compound_node<LHS,OP,RHS> >
      {
        private:
          typedef typename expression_type<LHS>::Result LHS_Result;
          typedef typename expression_type<RHS>::Result RHS_Result;
          
        public:
          typedef typename expression_type<compound_node<LHS_Result, OP, RHS_Result> >::Result Result;
      };

      /*
       * Compound Nodes - usual operators
       */
      template <class LHS, class LHS_SIZE_DESCRIPTOR ,class OP ,class RHS, class RHS_SIZE_DESCRIPTOR>
      struct expression_type<compound_node<vector_expression<LHS,LHS_SIZE_DESCRIPTOR>,
                                           OP,
                                           vector_expression<RHS,RHS_SIZE_DESCRIPTOR> >
                            >
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
          
        public:
          typedef vector_expression<T, LHS_SIZE_DESCRIPTOR> Result;
      };


      template <class LHS, class LHS_SIZE1_DESCRIPTOR, class LHS_SIZE2_DESCRIPTOR,
                class OP,
                class RHS, class RHS_SIZE1_DESCRIPTOR, class RHS_SIZE2_DESCRIPTOR>
      struct expression_type<compound_node<matrix_expression<LHS, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR>,
                                           OP,
                                           matrix_expression<RHS, RHS_SIZE1_DESCRIPTOR, RHS_SIZE2_DESCRIPTOR> >
                             > 
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
          
        public:
          typedef matrix_expression<T, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR> Result;
      };

      template <class LHS, class OP, class RHS>
      struct expression_type<compound_node<scalar_expression<LHS>, 
                                           OP,
                                           scalar_expression<RHS> >
                            > 
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
          
        public:
          typedef scalar_expression<T> Result;
      };

      /*
       * Scalar Operators
       */
      template <class LHS, class LHS_SIZE_DESCRIPTOR,
                class OP,
                class RHS>
      struct  expression_type<compound_node<vector_expression<LHS,LHS_SIZE_DESCRIPTOR>,
                                            OP,
                                            scalar_expression<RHS> > >
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
        public:
          typedef vector_expression<T, LHS_SIZE_DESCRIPTOR> Result;
      };

      template <class LHS,
                class OP,
                class RHS, class RHS_SIZE_DESCRIPTOR>
      struct expression_type<compound_node<scalar_expression<LHS>,
                                           OP,
                                           vector_expression<RHS,RHS_SIZE_DESCRIPTOR> >
                            > 
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
        public:
          typedef vector_expression<T, RHS_SIZE_DESCRIPTOR> Result;
      };


      template <class LHS, class LHS_SIZE1_DESCRIPTOR, class LHS_SIZE2_DESCRIPTOR,
                class OP,
                class RHS>
      struct expression_type<compound_node<matrix_expression<LHS,LHS_SIZE1_DESCRIPTOR,LHS_SIZE2_DESCRIPTOR>,
                                           OP,
                                           scalar_expression<RHS> >
                            > 
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
        public:
          typedef matrix_expression<T, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR> Result;
      };

      template <class LHS, 
                class OP,
                class RHS, class RHS_SIZE1_DESCRIPTOR, class RHS_SIZE2_DESCRIPTOR>
      struct expression_type<compound_node<scalar_expression<LHS>,
                                           OP,
                                           matrix_expression<RHS,RHS_SIZE1_DESCRIPTOR, RHS_SIZE2_DESCRIPTOR> >
                            >
      {
        private:
          typedef compound_node<LHS ,OP, RHS> T;
        public:
          typedef matrix_expression<T, RHS_SIZE1_DESCRIPTOR, RHS_SIZE2_DESCRIPTOR> Result;
      };


      /*
       * Compound Nodes - Non Trivial Operators
       */

      //Matrix-Vector product
      template <class LHS, class LHS_SIZE1_DESCRIPTOR, class LHS_SIZE2_DESCRIPTOR,
                class RHS, class RHS_SIZE_DESCRIPTOR>
      struct expression_type<compound_node<matrix_expression<LHS, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR>,
                                           prod_type,
                                           vector_expression<RHS,RHS_SIZE_DESCRIPTOR> >
                            >
      {
        typedef vector_expression<compound_node<LHS,prod_type,RHS>, LHS_SIZE1_DESCRIPTOR > Result;
      };

      template <class LHS, class LHS_SIZE1_DESCRIPTOR, class LHS_SIZE2_DESCRIPTOR,
                class RHS>
      struct expression_type<compound_node<matrix_expression<LHS, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR>,
                                           prod_type,
                                           constant_expression<RHS> >
                            >
      {
        typedef vector_expression<compound_node<LHS,prod_type,RHS>, LHS_SIZE1_DESCRIPTOR > Result;
      };


      template <class T>
      struct expression_type<inner_prod_impl_t<T> >
      {
        typedef scalar_expression<T> Result;
      };

      //Matrix-Matrix product
      template <class LHS, class LHS_SIZE1_DESCRIPTOR, class LHS_SIZE2_DESCRIPTOR,
                class RHS, class RHS_SIZE1_DESCRIPTOR, class RHS_SIZE2_DESCRIPTOR>
      struct expression_type<compound_node<matrix_expression<LHS, LHS_SIZE1_DESCRIPTOR, LHS_SIZE2_DESCRIPTOR>,
                                           prod_type,
                                           matrix_expression<RHS,RHS_SIZE1_DESCRIPTOR,RHS_SIZE2_DESCRIPTOR> >
                            >
      {
        typedef matrix_expression<compound_node<LHS,prod_type,RHS>, LHS_SIZE1_DESCRIPTOR, RHS_SIZE2_DESCRIPTOR > Result;
      };

      //Inner product
      template <class LHS, class LHS_SIZE_DESCRIPTOR,
                class RHS, class RHS_SIZE_DESCRIPTOR>
      struct expression_type< compound_node<vector_expression<LHS,LHS_SIZE_DESCRIPTOR>,
                                            inner_prod_type,
                                            vector_expression<RHS,RHS_SIZE_DESCRIPTOR> >
                            >
      {
        typedef scalar_expression<compound_node<LHS,inner_prod_type,RHS> > Result;
      };

      template <class LHS, class LHS_SIZE_DESCRIPTOR,
                class RHS>
      struct expression_type< compound_node<vector_expression<LHS,LHS_SIZE_DESCRIPTOR>,
                                            inner_prod_type,
                                             constant_expression<RHS> >
                            >
      {
        typedef scalar_expression<compound_node<LHS,inner_prod_type,RHS> > Result;
      };


      template<class OP, class RHS>
      struct expression_type<compound_node<NullType,OP,RHS> >
      {
        typedef typename expression_type<RHS>::Result Result;
      };

      template<class LHS, class OP>
      struct expression_type<compound_node<LHS,OP,NullType> >
      {
        typedef typename expression_type<LHS>::Result Result;
      };


      /*
       * Elementwise Modifiers
       */
      template <class T>
      struct expression_type< elementwise_modifier<T> >
      {
        typedef typename expression_type<typename T::PRIOR_TYPE>::Result Result;
      };

      template <class T, class SIZE_DESCRIPTOR>
      struct expression_type< vector_expression<T,SIZE_DESCRIPTOR> > 
      {
        typedef typename expression_type<T>::Result Result;
      };

      template <class T, class SIZE1_DESCRIPTOR, class SIZE2_DESCRIPTOR>
      struct expression_type< matrix_expression<T,SIZE1_DESCRIPTOR,SIZE2_DESCRIPTOR> > 
      {
        typedef typename expression_type<T>::Result Result;
      };

      template <class T>
      struct expression_type< scalar_expression<T> > 
      {
        typedef typename expression_type<T>::Result Result;
      };

      /*
        * Symbolic Constant
        */
      template <long VALUE>
      struct expression_type<symbolic_constant<VALUE> >
      {
        typedef constant_expression<symbolic_constant<VALUE> > Result;
      };

      /*
       * Symbolic Vectors
       */

      template <unsigned int ID,typename SCALARTYPE, unsigned int ALIGNMENT>
      struct expression_type< symbolic_vector<ID,SCALARTYPE,ALIGNMENT> > 
      {
        typedef vector_expression<symbolic_vector<ID,SCALARTYPE,ALIGNMENT>,
                                  symbolic_vector<ID,SCALARTYPE,ALIGNMENT> > Result;
      };

      /*
       * Symbolic Matrices
       */

      template <unsigned int ID,typename SCALARTYPE, class F, unsigned int ALIGNMENT>
      struct expression_type<symbolic_matrix<ID,SCALARTYPE,F,ALIGNMENT> >
      {
        private:
          typedef symbolic_matrix<ID,SCALARTYPE,F,ALIGNMENT> T;
        public:
          typedef matrix_expression<T, T, T> Result;
      };

      /*
       * Symbolic Scalars
       */

      template <unsigned int ID, typename SCALARTYPE>
      struct expression_type<cpu_symbolic_scalar<ID, SCALARTYPE> > 
      {
        typedef scalar_expression<cpu_symbolic_scalar<ID, SCALARTYPE> > Result;
      };

      template <unsigned int ID, typename SCALARTYPE>
      struct expression_type<gpu_symbolic_scalar<ID, SCALARTYPE> > 
      {
        typedef scalar_expression< gpu_symbolic_scalar<ID, SCALARTYPE> > Result;
      };

    }//result_of

    /*
     * Traits
     */

    namespace result_of
    {

      template<class T>
      struct is_symbolic_vector
      {
        enum { value = 0 };
      };

      template<unsigned int Id, class ScalarType, unsigned int Alignment>
      struct is_symbolic_vector<symbolic_vector<Id,ScalarType,Alignment> >
      {
        enum { value = 1 };
      };


      template<class T>
      struct is_symbolic_matrix
      {
        enum { value = 0 };
      };

      template<unsigned int Id, class ScalarType, class Layout, unsigned int Alignment>
      struct is_symbolic_matrix<symbolic_matrix<Id,ScalarType,Layout,Alignment> >
      {
        enum { value = 1 };
      };

      template<class T>
      struct is_symbolic_cpu_scalar
      {
        enum { value = 0 };
      };

      template<unsigned int Id, class ScalarType>
      struct is_symbolic_cpu_scalar<cpu_symbolic_scalar<Id,ScalarType> >
      {
        enum { value = 1 };
      };

      template<class T>
      struct is_symbolic_gpu_scalar
      {
        enum { value = 0 };
      };

      template<unsigned int Id, class ScalarType>
      struct is_symbolic_gpu_scalar<gpu_symbolic_scalar<Id,ScalarType> >
      {
        enum { value = 1 };
      };

      template <class T>
      struct is_row_major
      {
        enum { value = 0 };
      };

      template <unsigned int ID, class ScalarType,  unsigned int Alignment>
      struct is_row_major<symbolic_matrix<ID, ScalarType, viennacl::row_major, Alignment> >
      {
        enum { value = 1 };
      };

      template <class T>
      struct is_transposed
      {
        enum { value = 0 };
      };

      template <class T>
      struct is_kernel_argument
      {
        enum { value = 0 };
      };

      template <unsigned int ID,class SCALARTYPE, unsigned int ALIGNMENT>
      struct is_kernel_argument<symbolic_vector<ID,SCALARTYPE,ALIGNMENT> >
      {
        enum { value = 1 };
      };

      template <unsigned int ID,class SCALARTYPE, class F, unsigned int ALIGNMENT>
      struct is_kernel_argument<symbolic_matrix<ID,SCALARTYPE,F,ALIGNMENT> >
      {
        enum { value = 1 };
      };

      template <unsigned int ID, class SCALARTYPE>
      struct is_kernel_argument<cpu_symbolic_scalar<ID, SCALARTYPE> >
      {
        enum { value = 1 };
      };

      template <unsigned int ID, class SCALARTYPE>
      struct is_kernel_argument<gpu_symbolic_scalar<ID, SCALARTYPE> >
      {
        enum { value = 1 };
      };

      template<class T>
      struct is_kernel_argument<inner_prod_impl_t<T> >
      {
        enum { value = 1 };
      };

      template<class LHS, class RHS>
      struct is_kernel_argument<compound_node<LHS,inner_prod_type,RHS> >
      {
        enum { value = 1 };
      };

      template<class Bound, class Expr>
      struct is_kernel_argument< repeater_impl<Bound, Expr> >
      {
        enum { value = 1 };
      };

      template <class T>
      struct is_inner_product_leaf
      {
        enum { value = 0};
      };

      template <class LHS,class RHS>
      struct is_inner_product_leaf<compound_node<LHS,inner_prod_type,RHS> >
      {
        enum { value = 1};
      };


      template <class T>
      struct is_product_leaf
      {
        enum { value = 0};
      };

      template <class LHS,class RHS>
      struct is_product_leaf<compound_node<LHS,prod_type,RHS> >
      {
        enum { value = 1};
      };

      template <class LHS,class RHS>
      struct is_product_leaf<compound_node<LHS,scal_mul_type,RHS> >
      {
        enum { value = result_of::is_product_leaf<RHS>::value || result_of::is_product_leaf<LHS>::value };
      };

      template <class T>
      struct is_null_type
      {
        enum { value = 0 };
      };

      template <>
      struct is_null_type<NullType>
      {
        enum { value = 1 };
      };

      template <class T>
      struct is_compound
      {
        enum { value = 0 } ;
      };

      template <class LHS, class OP, class RHS>
      struct is_compound<compound_node<LHS,OP,RHS> >
      {
        enum {value = 1};
      };

      template<class T>
      struct is_inner_product_impl
      {
        enum { value = 0 };
      };

      template<class T>
      struct is_inner_product_impl<inner_prod_impl_t<T> >
      {
        enum { value = 1 };
      };

      template<class T>
      struct is_symbolic_constant
      {
        enum { value = 0};
      };

      template<class T>
      struct is_symbolic_expression
      {
        enum { value =   is_symbolic_vector<T>::value
                      || is_symbolic_matrix<T>::value
                      || is_compound<T>::value
                      || is_symbolic_cpu_scalar<T>::value
                      || is_symbolic_gpu_scalar<T>::value };
      };

      template <class T>
      struct is_scalar_expression_impl
      {
        enum { value = 0 };
      };

      template <class T>
      struct is_scalar_expression_impl<result_of::scalar_expression<T> >
      {
        enum { value = 1};
      };

      template <class T>
      struct is_scalar_expression
      {
        enum { value = is_scalar_expression_impl<typename result_of::expression_type<T>::Result >::value };
      };


      template <class T>
      struct is_vector_expression_impl
      {
        enum { value = 0 };
      };

      template <class T, class SIZE_D>
      struct is_vector_expression_impl<result_of::vector_expression<T,SIZE_D> >
      {
        enum { value = 1};
      };

      template <class T>
      struct is_vector_expression
      {
        enum { value = is_vector_expression_impl<typename result_of::expression_type<T>::Result >::value };
      };

      template <class T>
      struct is_matrix_expression_impl
      {
        enum { value = 0 };
      };

      template <class T, class SIZE1_D,class SIZE2_D>
      struct is_matrix_expression_impl<result_of::matrix_expression<T,SIZE1_D, SIZE2_D> >
      {
        enum { value = 1};
      };

      template <class T>
      struct is_matrix_expression
      {
        enum { value = is_matrix_expression_impl<typename result_of::expression_type<T>::Result >::value };
      };

      template <class EXPR1, class EXPR2>
      struct is_same_expression_type
      {
        enum { value = (is_vector_expression<EXPR1>::value && is_vector_expression<EXPR2>::value)
                      || (is_matrix_expression<EXPR1>::value && is_matrix_expression<EXPR2>::value)
                      || (is_scalar_expression<EXPR1>::value && is_scalar_expression<EXPR2>::value) };
      };


      /** @brief Special case: symbolic constant for elementwise can be used as every type. */
      template<class Expr, long VAL>
      struct is_same_expression_type<Expr,symbolic_constant<VAL> >
      {
        enum { value = 1 };
      };

      /** @brief Special case: symbolic constant for elementwise can be used as every type. */
      template<class Expr, long VAL>
      struct is_same_expression_type<symbolic_constant<VAL>, Expr>
      {
        enum { value = 1 };
      };

    }

  }//generator

}//ViennaCL

#endif


