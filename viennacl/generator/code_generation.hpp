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
          enum operation_type{
            none = 0,
            saxpy_vector = 1,
            saxpy_matrix = 2,
            scalar_reduction = 3,
            vector_reduction = 4,
            matrix_product = 5
          };


        public:

          operations_handler(){
            last_operation_ = none;
          }

          //SAXPY VECTOR
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_saxpy_vector_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             if(last_operation_!=saxpy_vector){
               operations_.push_back(tools::shared_ptr<saxpy_vector_generator>(new saxpy_vector_generator(new saxpy_vector_profile())));
             }
             static_cast<saxpy_vector_generator*>(operations_.back().get())->add_expression(new T(op));
             last_operation_=saxpy_vector;
          }

          //SAXPY MATRIX
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_saxpy_matrix_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             if(last_operation_!=saxpy_matrix)
               operations_.push_back(tools::shared_ptr<saxpy_matrix_generator>(new saxpy_matrix_generator(new saxpy_matrix_profile())));
             static_cast<saxpy_matrix_generator*>(operations_.back().get())->add_expression(new T(op));
             last_operation_=saxpy_matrix;
          }

          //SCALAR REDUCTION
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_scalar_reduction_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             if(last_operation_!=scalar_reduction)
               operations_.push_back(tools::shared_ptr<scalar_reduction_generator>(new scalar_reduction_generator(new scalar_reduction_profile())));
             static_cast<scalar_reduction_generator*>(operations_.back().get())->add_expression(new T(op));
             last_operation_=scalar_reduction;
          }

          //VECTOR REDUCTION
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_vector_reduction_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             operations_.push_back(tools::shared_ptr<vector_reduction_generator>(new vector_reduction_generator(new vector_reduction_profile())));
             static_cast<vector_reduction_generator*>(operations_.back().get())->add_expression(new T(op));
             last_operation_ = vector_reduction;
          }

          //MATRIX PRODUCT
          template<class T>
          typename viennacl::enable_if<viennacl::generator::result_of::is_matrix_product_operation<T>::value,void>::type
          add(T const & op){
             representation_ << typeid(T).name();
             operations_.push_back(tools::shared_ptr<matrix_product_generator>(new matrix_product_generator(new matrix_product_profile())));
             static_cast<matrix_product_generator*>(operations_.back().get())->add_expression(new T(op));
             last_operation_ = matrix_product;
          }

          std::string representation(){
            return representation_.str();
          }

          void enqueue(viennacl::ocl::program & pgm){
            unsigned int n=0;
            for(std::list<tools::shared_ptr<generator_base> >::iterator it = operations_.begin() ; it != operations_.end() ; ++it){
              (*it)->enqueue(n, pgm);
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
            for(std::list<tools::shared_ptr<generator_base> >::const_iterator it = operations_.begin() ; it != operations_.end() ; ++it){
              (*it)->generate(kernel_counter,kss);
            }
            return oss.str();
          }

        private:
          std::list< tools::shared_ptr<generator_base> > operations_;
          operation_type last_operation_;
          std::ostringstream representation_;
      };

    }


  }

}
#endif // KERNEL_GENERATOR_FRONTEND_HPP
