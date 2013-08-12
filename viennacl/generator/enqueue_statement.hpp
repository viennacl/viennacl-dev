#ifndef VIENNACL_GENERATOR_ENQUEUE_TREE_HPP
#define VIENNACL_GENERATOR_ENQUEUE_TREE_HPP

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


/** @file viennacl/generator/enqueue_tree.hpp
    @brief Functor to enqueue the leaves of an expression tree
*/

#include <set>

#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/generator/forwards.h"

#include "viennacl/tools/shared_ptr.hpp"

#include "viennacl/ocl/kernel.hpp"

#include "viennacl/generator/generate_utils.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/mapped_types.hpp"


namespace viennacl{

  namespace generator{

    namespace detail{

      class enqueue_functor : public traversal_functor{
        public:
          typedef void result_type;

          enqueue_functor(std::set<void *> & memory, unsigned int & current_arg, viennacl::ocl::kernel & kernel) : memory_(memory), current_arg_(current_arg), kernel_(kernel){ }

          template<class ScalarType>
          result_type operator()(ScalarType const & scal) const {
            kernel_.arg(current_arg_++, scal);
          }

          //Scalar mapping
          template<class ScalarType>
          result_type operator()(scalar<ScalarType> const & scal) const {
            if(memory_.insert((void*)&scal).second)
              kernel_.arg(current_arg_++, scal.handle().opencl_handle());
          }

          //Vector mapping
          template<class ScalarType>
          result_type operator()(vector_base<ScalarType> const & vec) const {
            if(memory_.insert((void*)&vec).second){
              kernel_.arg(current_arg_++, vec.handle().opencl_handle());
              if(viennacl::traits::start(vec)>0)
                kernel_.arg(current_arg_++, viennacl::traits::start(vec));
              if(vec.stride()>1)
                kernel_.arg(current_arg_++, vec.stride());
            }
          }

          //Implicit vector mapping
          template<class ScalarType>
          result_type operator()(implicit_vector_base<ScalarType> const & vec) const {
            if(memory_.insert((void*)&vec).second){
              if(vec.is_value_static()==false)
                kernel_.arg(current_arg_++, vec.value());
              if(vec.has_index())
                kernel_.arg(current_arg_++, vec.index());
            }
          }

          //Matrix mapping
          template<class ScalarType, class Layout>
          result_type operator()(matrix_base<ScalarType, Layout> const & mat) const {
            if(memory_.insert((void*)&mat).second){
              kernel_.arg(current_arg_++, mat.handle().opencl_handle());
              if(viennacl::traits::start1(mat)>0)
                kernel_.arg(current_arg_++, viennacl::traits::start1(mat));
              if(viennacl::traits::stride1(mat)>1)
                kernel_.arg(current_arg_++, viennacl::traits::stride1(mat));
              if(viennacl::traits::start2(mat)>0)
                kernel_.arg(current_arg_++, viennacl::traits::start2(mat));
              if(viennacl::traits::stride2(mat)>1)
                kernel_.arg(current_arg_++, viennacl::traits::stride2(mat));
            }
          }

          //Implicit matrix mapping
          template<class ScalarType>
          result_type operator()(implicit_matrix_base<ScalarType> const & mat) const {
            if(mat.is_value_static()==false)
              kernel_.arg(current_arg_++, mat.value());
          }

          //Traversal functor:
          void operator()(scheduler::statement const * /*statement*/, scheduler::statement_node const * root_node, detail::node_type node_type) const {
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
