#ifndef VIENNACL_GENERATOR_CODE_GENERATION_HPP
#define VIENNACL_GENERATOR_CODE_GENERATION_HPP

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


/** @file viennacl/generator/code_generation.hpp
 *
 * Functions related to the code generation itself
*/

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include <typeinfo>

#include "viennacl/ocl/infos.hpp"


#include "viennacl/generator/templates/matrix_product.hpp"
#include "viennacl/generator/templates/vector_reduction.hpp"
#include "viennacl/generator/templates/scalar_reduction.hpp"
#include "viennacl/generator/templates/saxpy_vector.hpp"
#include "viennacl/generator/templates/saxpy_matrix.hpp"

#include "viennacl/generator/symbolic_types.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/builtin_database.hpp"
#include "viennacl/generator/traits.hpp"

#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl{

  namespace generator{

    namespace code_generation{

      /** @brief Handler for the operations
       *
       * Essentially segments the kernels and handles the generation of the program source code
       */
      class operations_handler{
        private:
          profile_base const * get_profile(code_generation::profile_id const & key) const {
            //Check forced profiles
            {
              std::map< profile_id, tools::shared_ptr<profile_base> >::const_iterator it = forced_profiles_.find(key);
              if(it!=forced_profiles_.end())
                return it->second.get();
            }

            //Fallback on builtin database
            {
              cl_device_id id = viennacl::ocl::current_device().id();
              code_generation::builtin_database_t::iterator it = code_generation::builtin_database.find(std::make_pair(ocl::info<CL_DEVICE_VENDOR_ID>(id), ocl::info<CL_DEVICE_TYPE>(id)));
              if(it!=code_generation::builtin_database.end()){
                code_generation::builtin_database_t::value_type::second_type::iterator it2 = it->second.find(key);
                if(it2!=it->second.end())
                  return it2->second;
              }
            }

            //Fallback on default
            return code_generation::default_profiles.at(key.first);
          }

        public:

          template<class PROF>
          void force_profile(code_generation::profile_id const & id, PROF const & prof){
            forced_profiles_.insert(std::make_pair(id, tools::shared_ptr<profile_base>(new PROF(prof))));
          }

          //SAXPY VECTOR
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_saxpy_vector_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             code_generation::profile_id new_operation = std::make_pair(code_generation::axpy, sizeof(typename T::ScalarType));
             if(new_operation!=last_operation_){
               operations_.push_back(std::make_pair(tools::shared_ptr<saxpy_vector_generator>(new saxpy_vector_generator()),new_operation));
               last_operation_=new_operation;
             }
             static_cast<saxpy_vector_generator*>(operations_.back().first.get())->add_expression(new T(op));
          }

          //SAXPY MATRIX
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_saxpy_matrix_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             code_generation::profile_id new_operation = std::make_pair(code_generation::aXpY, sizeof(typename T::ScalarType));
             if(new_operation!=last_operation_){
               operations_.push_back(std::make_pair(tools::shared_ptr<saxpy_matrix_generator>(new saxpy_matrix_generator()),new_operation));
               last_operation_=new_operation;
             }
             static_cast<saxpy_matrix_generator*>(operations_.back().first.get())->add_expression(new T(op));
          }

          //SCALAR REDUCTION
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_scalar_reduction_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             code_generation::profile_id new_operation = std::make_pair(code_generation::dot, sizeof(typename T::ScalarType));
             if(new_operation!=last_operation_){
               operations_.push_back(std::make_pair(tools::shared_ptr<scalar_reduction_generator>(new scalar_reduction_generator()),new_operation));
               last_operation_=new_operation;
             }
             static_cast<scalar_reduction_generator*>(operations_.back().first.get())->add_expression(new T(op));
          }

          //VECTOR REDUCTION
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_vector_reduction_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             code_generation::profile_id new_operation;
             bool is_lhs_transposed = result_of::is_transposed<typename T::Rhs::Lhs>::value;
             bool is_lhs_row_major = result_of::is_row_major<typename T::Rhs::Lhs>::value;
             if(is_lhs_transposed){
               if(is_lhs_row_major)
                 new_operation.first = code_generation::gemvTv;
               else
                 new_operation.first = code_generation::gemvAv;
             }
             else{
               if(is_lhs_row_major)
                 new_operation.first = code_generation::gemvAv;
               else
                 new_operation.first = code_generation::gemvTv;
             }
             new_operation.second = sizeof(typename T::ScalarType);
             if(new_operation!=last_operation_){
               operations_.push_back(std::make_pair(tools::shared_ptr<vector_reduction_generator>(new vector_reduction_generator()),new_operation));
               last_operation_=new_operation;
             }
             static_cast<vector_reduction_generator*>(operations_.back().first.get())->add_expression(new T(op));
          }

          //MATRIX PRODUCT
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_matrix_product_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             code_generation::profile_id new_operation;
             bool is_lhs_transposed = result_of::is_transposed<typename T::Rhs::Lhs>::value;
             bool is_lhs_row_major = result_of::is_row_major<typename T::Rhs::Lhs>::value;
             bool is_rhs_transposed = result_of::is_transposed<typename T::Rhs::Rhs>::value;
             bool is_rhs_row_major = result_of::is_row_major<typename T::Rhs::Rhs>::value;

             if(is_lhs_transposed)
               if(is_lhs_row_major)
                 if(is_rhs_transposed)
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmTT;
                   else
                     new_operation.first = code_generation::gemmTA;
                 else
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmTA;
                   else
                     new_operation.first = code_generation::gemmTT;
               else
                 if(is_rhs_transposed)
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmAT;
                   else
                     new_operation.first = code_generation::gemmAA;
                 else
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmAA;
                   else
                     new_operation.first = code_generation::gemmAT;
             else
               if(is_lhs_row_major)
                 if(is_rhs_transposed)
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmAT;
                   else
                     new_operation.first = code_generation::gemmAA;
                 else
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmAA;
                   else
                     new_operation.first = code_generation::gemmAT;
               else
                 if(is_rhs_transposed)
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmTT;
                   else
                     new_operation.first = code_generation::gemmTA;
                 else
                   if(is_rhs_row_major)
                     new_operation.first = code_generation::gemmTA;
                   else
                     new_operation.first = code_generation::gemmTT;

             new_operation.second = sizeof(typename T::ScalarType);
             if(new_operation!=last_operation_){
               operations_.push_back(std::make_pair(tools::shared_ptr<matrix_product_generator>(new matrix_product_generator()),new_operation));
               last_operation_=new_operation;
             }
             static_cast<matrix_product_generator*>(operations_.back().first.get())->add_expression(new T(op));
          }



          std::string representation(){
            return representation_.str();
          }

          void enqueue(viennacl::ocl::program & pgm){
            unsigned int n=0;
            for(std::list< std::pair<tools::shared_ptr<generator_base>, code_generation::profile_id > >::iterator it = operations_.begin() ; it != operations_.end() ; ++it){
              it->first->set_profile(get_profile(it->second));
              it->first->enqueue(n, pgm);
            }
          }

          std::string get_source_code() const{
            std::ostringstream oss;
            utils::kernel_generation_stream kss(oss);
            kss << "#if defined(cl_khr_fp64)\n";
            kss <<  "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
            kss <<  "#elif defined(cl_amd_fp64)\n";
            kss <<  "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n";
            kss <<  "#endif\n";
            unsigned int kernel_counter = 0;
            for(std::list< std::pair<tools::shared_ptr<generator_base>, code_generation::profile_id > >::const_iterator it = operations_.begin() ; it != operations_.end() ; ++it){
              it->first->set_profile(get_profile(it->second));
              it->first->generate(kernel_counter,kss);
            }
            return oss.str();
          }

        private:
          std::list< std::pair<tools::shared_ptr<generator_base>, code_generation::profile_id > > operations_;
           std::map< profile_id, tools::shared_ptr<profile_base> > forced_profiles_;
          code_generation::profile_id last_operation_;
          std::ostringstream representation_;
      };

    }


  }

}
#endif // KERNEL_GENERATOR_FRONTEND_HPP
