#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_SET_ARGUMENTS_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_SET_ARGUMENTS_HPP

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


/** @file viennacl/generator/set_arguments_functor.hpp
    @brief Functor to set the arguments of a statement into a kernel
*/

#include <set>

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"

#include "viennacl/meta/result_of.hpp"

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/ocl/kernel.hpp"

#include "viennacl/device_specific/tree_parsing/traverse.hpp"
#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/mapped_objects.hpp"


namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      class set_arguments_functor : public traversal_functor{
        public:
          typedef void result_type;

          set_arguments_functor(std::set<void *> & memory, unsigned int & current_arg, viennacl::ocl::kernel & kernel) : memory_(memory), current_arg_(current_arg), kernel_(kernel){ }

          template<class ScalarType>
          result_type operator()(ScalarType const & scal) const {
            typedef typename viennacl::result_of::cl_type<ScalarType>::type cl_scalartype;
            kernel_.arg(current_arg_++, cl_scalartype(scal));
          }

          /** @brief Scalar mapping */
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const {
            if(memory_.insert((void*)&scal).second)
              kernel_.arg(current_arg_++, scal.handle().opencl_handle());
          }

          /** @brief Vector mapping */
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const {
            if(memory_.insert((void*)&vec).second){
              kernel_.arg(current_arg_++, vec.handle().opencl_handle());
              kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start(vec)));
              kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride(vec)));
            }
          }

          /** @brief Implicit vector mapping */
          template<class ScalarType>
          result_type operator()(implicit_vector_base<ScalarType> const & vec) const {
            typedef typename viennacl::result_of::cl_type<ScalarType>::type cl_scalartype;
            if(memory_.insert((void*)&vec).second){
              if(vec.is_value_static()==false)
                kernel_.arg(current_arg_++, cl_scalartype(vec.value()));
              if(vec.has_index())
                kernel_.arg(current_arg_++, cl_uint(vec.index()));
            }
          }

          /** @brief Matrix mapping */
          template<class ScalarType>
          result_type operator()(matrix_base<ScalarType> const & mat) const {
            if(memory_.insert((void*)&mat).second){
              kernel_.arg(current_arg_++, mat.handle().opencl_handle());
              if(mat.row_major()){
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::internal_size2(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start2(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride2(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start1(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride1(mat)));
              }
              else{
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::internal_size1(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start1(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride1(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::start2(mat)));
                kernel_.arg(current_arg_++, cl_uint(viennacl::traits::stride2(mat)));
              }
            }
          }

          /** @brief Implicit matrix mapping */
          template<class ScalarType>
          result_type operator()(implicit_matrix_base<ScalarType> const & mat) const {
            if(mat.is_value_static()==false)
              kernel_.arg(current_arg_++, mat.value());
          }

          /** @brief Traversal functor: */
          void operator()(scheduler::statement const * /*statement*/, scheduler::statement_node const * root_node, node_type node_type) const {
            if(node_type==LHS_NODE_TYPE && root_node->lhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node->lhs, *this);
            else if(node_type==RHS_NODE_TYPE && root_node->rhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node->rhs, *this);
          }

        private:
          std::set<void *> & memory_;
          unsigned int & current_arg_;
          viennacl::ocl::kernel & kernel_;
      };

    }

  }

}
#endif
