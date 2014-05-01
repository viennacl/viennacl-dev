#ifndef VIENNACL_DEVICE_SPECIFIC_FORWARDS_H
#define VIENNACL_DEVICE_SPECIFIC_FORWARDS_H

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


/** @file viennacl/generator/forwards.h
    @brief Forwards declaration
*/

#include <list>
#include <map>
#include <set>
#include <stdexcept>

#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace device_specific{

    static void generate_enqueue_statement(viennacl::scheduler::statement const & s, scheduler::statement_node const & root_node);
    static void generate_enqueue_statement(viennacl::scheduler::statement const & s);

    enum expression_type_family{
      SCALAR_SAXPY_FAMILY,
      VECTOR_SAXPY_FAMILY,
      MATRIX_SAXPY_FAMILY,
      SCALAR_REDUCE_FAMILY,
      VECTOR_REDUCE_FAMILY,
      MATRIX_PRODUCT_FAMILY,
      INVALID_EXPRESSION_FAMILY
    };

    inline bool is_scalar_reduction(scheduler::statement_node const & node){
      return node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE || node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY;
    }

    inline bool is_vector_reduction(scheduler::statement_node const & node){
      return node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE
          || node.op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY
          || node.op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY;
    }

    enum expression_type{
      SCALAR_SAXPY_TYPE,
      VECTOR_SAXPY_TYPE,
      MATRIX_SAXPY_TYPE,
      SCALAR_REDUCE_TYPE,
      VECTOR_REDUCE_Nx_TYPE,
      VECTOR_REDUCE_Tx_TYPE,
      MATRIX_PRODUCT_NN_TYPE,
      MATRIX_PRODUCT_TN_TYPE,
      MATRIX_PRODUCT_NT_TYPE,
      MATRIX_PRODUCT_TT_TYPE,
      INVALID_EXPRESSION_TYPE
    };

    inline const char * expression_type_to_string(expression_type type){
      switch(type){
        case SCALAR_SAXPY_TYPE : return "Scalar SAXPY";
        case VECTOR_SAXPY_TYPE : return "Vector SAXPY";
        case MATRIX_SAXPY_TYPE : return "Matrix SAXPY";
        case SCALAR_REDUCE_TYPE : return "Inner Product";
        case VECTOR_REDUCE_Nx_TYPE : return "Matrix-Vector Product : Ax";
        case VECTOR_REDUCE_Tx_TYPE : return "Matrix-Vector Product : Tx";
        case MATRIX_PRODUCT_NN_TYPE : return "Matrix-Matrix Product : AA";
        case MATRIX_PRODUCT_TN_TYPE : return "Matrix-Matrix Product : TA";
        case MATRIX_PRODUCT_NT_TYPE : return "Matrix-Matrix Product : AT";
        case MATRIX_PRODUCT_TT_TYPE : return "Matrix-Matrix Product : TT";
        default : return "INVALID EXPRESSION";
      }
    }

    /** @brief generate the string for a pointer kernel argument */
      static std::string generate_value_kernel_argument(std::string const & scalartype, std::string const & name){
        return scalartype + ' ' + name + ",";
      }

      /** @brief generate the string for a pointer kernel argument */
      static std::string generate_pointer_kernel_argument(std::string const & address_space, std::string const & scalartype, std::string const & name){
        return address_space +  " " + scalartype + "* " + name + ",";
      }

    /** @brief Emulation of C++11's .at() member for std::map<> */
    template <typename KeyT, typename ValueT>
    ValueT const & at(std::map<KeyT, ValueT> const & map, KeyT const & key)
    {
      typename std::map<KeyT, ValueT>::const_iterator it = map.find(key);
      if (it != map.end())
        return it->second;

      throw std::out_of_range("Generator: Key not found in map");
    }

    typedef std::pair<expression_type, std::size_t> expression_key_type;

    struct expression_descriptor{
        expression_key_type make_key() const { return expression_key_type(type,scalartype_size); }
        bool operator==(expression_descriptor const & other) const
        {
          return type_family == other.type_family && type == other.type && scalartype_size==other.scalartype_size;
        }
        expression_type_family type_family;
        expression_type type;
        std::size_t scalartype_size;
    };

    /** @brief Exception for the case the generator is unable to deal with the operation */
    class generator_not_supported_exception : public std::exception
    {
    public:
      generator_not_supported_exception() : message_() {}
      generator_not_supported_exception(std::string message) : message_("ViennaCL: Internal error: The scheduler encountered a problem with the operation provided: " + message) {}

      virtual const char* what() const throw() { return message_.c_str(); }

      virtual ~generator_not_supported_exception() throw() {}
    private:
      std::string message_;
    };



    namespace utils{
      class kernel_generation_stream;
    }


    namespace tree_parsing{
        enum node_type{
          LHS_NODE_TYPE,
          PARENT_NODE_TYPE,
          RHS_NODE_TYPE
        };
    }

    class mapped_object;

    typedef std::pair<scheduler::statement_node const *, tree_parsing::node_type> key_type;
    typedef tools::shared_ptr<mapped_object> container_ptr_type;
    typedef std::map<key_type, container_ptr_type> mapping_type;

    static std::string generate(std::pair<std::string, std::string> const & index, int vector_index, mapped_object const & s);
    static void fetch(std::pair<std::string, std::string> const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream, mapped_object & s);
    static const char * generate(scheduler::operation_node_type arg);
    static std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str, mapped_object const & s);

    namespace tree_parsing{

      template<class Fun>
      static void traverse(scheduler::statement const & statement, scheduler::statement_node const & root_node, Fun const & fun, bool recurse_binary_leaf = true);
      static void generate_all_rhs(scheduler::statement const & statement
                                , scheduler::statement_node const & root_node
                                , std::pair<std::string, std::string> const & index
                                , int vector_element
                                , std::string & str
                                , mapping_type const & mapping);

      struct map_functor;

    }

  }

}
#endif
