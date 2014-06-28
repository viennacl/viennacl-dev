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

#include "viennacl/ocl/forwards.h"
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/backend/mem_handle.hpp"

namespace viennacl{

  namespace device_specific{

    struct index_tuple{
      index_tuple(std::string const & _i, std::string const & _bound0) : i(_i), bound0(_bound0), j(""), bound1(""){ }
      index_tuple(std::string const & _i, std::string const & _bound0, std::string const & _j, std::string const & _bound1) : i(_i), bound0(_bound0), j(_j), bound1(_bound1){ }
      std::string i;
      std::string bound0;
      std::string j;
      std::string bound1;
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
      SCALAR_AXPY_TYPE,
      VECTOR_AXPY_TYPE,
      MATRIX_AXPY_TYPE,
      REDUCTION_TYPE,
      ROW_WISE_REDUCTION_Nx_TYPE,
      ROW_WISE_REDUCTION_Tx_TYPE,
      MATRIX_PRODUCT_NN_TYPE,
      MATRIX_PRODUCT_TN_TYPE,
      MATRIX_PRODUCT_NT_TYPE,
      MATRIX_PRODUCT_TT_TYPE,
      INVALID_EXPRESSION_TYPE
    };

    inline const char * expression_type_to_string(expression_type type){
      switch(type){
        case SCALAR_AXPY_TYPE : return "Scalar AXPY";
        case VECTOR_AXPY_TYPE : return "Vector AXPY";
        case MATRIX_AXPY_TYPE : return "Matrix AXPY";
        case REDUCTION_TYPE : return "Reduction";
        case ROW_WISE_REDUCTION_Nx_TYPE : return "Row-wise reduction: Ax";
        case ROW_WISE_REDUCTION_Tx_TYPE : return "Row-wise reduction : Tx";
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

    /** @brief Exception for the case the generator is unable to deal with the operation */
    class generator_not_supported_exception : public std::exception
    {
    public:
      generator_not_supported_exception() : message_() {}
      generator_not_supported_exception(std::string message) : message_("ViennaCL: Internal error: The generator cannot handle the statement provided: " + message) {}
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

    typedef std::map<std::pair<unsigned int, tree_parsing::node_type>, tools::shared_ptr<mapped_object> > mapping_type;


    namespace tree_parsing{

      template<class Fun>
      inline void traverse(scheduler::statement const & statement, unsigned int root_idx, Fun const & fun, bool inspect);

      inline std::string evaluate_expression(scheduler::statement const & statement, unsigned int root_idx, index_tuple const & index, int simd_element, mapping_type const & mapping, node_type initial_leaf);

      struct map_functor;

    }

    using scheduler::INT_TYPE;
    using scheduler::UINT_TYPE;
    using scheduler::ULONG_TYPE;
    using scheduler::LONG_TYPE;
    using scheduler::FLOAT_TYPE;
    using scheduler::DOUBLE_TYPE;

    typedef cl_uint vendor_id_type;
    typedef cl_device_type device_type;
    typedef std::string device_name_type;

    template<class ParamT>
    class database_type
    {
    public:
      typedef std::map<scheduler::statement_node_numeric_type, ParamT> expression_map;
      typedef std::map<device_name_type, expression_map> device_name_map;
      typedef std::map<ocl::device_architecture_family, device_name_map> device_architecture_map;
      typedef std::map<device_type, device_architecture_map> device_type_map;
      typedef std::map<vendor_id_type, device_type_map> map_type;
      map_type map;

      database_type(vendor_id_type p0, device_type p1, ocl::device_architecture_family p2, device_name_type p3, scheduler::statement_node_numeric_type p4, ParamT const & p5)
      {
        map[p0][p1][p2][p3].insert(std::make_pair(p4, p5));
      }

      database_type<ParamT> & operator()(vendor_id_type p0, device_type p1, ocl::device_architecture_family p2, device_name_type p3, scheduler::statement_node_numeric_type p4, ParamT const & p5)
      {
        map[p0][p1][p2][p3].insert(std::make_pair(p4, p5));
         return *this;
      }
    };

    namespace database{
      using scheduler::FLOAT_TYPE;
      using scheduler::DOUBLE_TYPE;
      using namespace viennacl::ocl;
    }

    class symbolic_binder{
    public:
      virtual ~symbolic_binder(){ }
      virtual bool bind(viennacl::backend::mem_handle const * ph) = 0;
      virtual unsigned int get(viennacl::backend::mem_handle const * ph) = 0;
    };

    class bind_to_handle : public symbolic_binder{
    public:
      bind_to_handle() : current_arg_(0){ }
      bool bind(viennacl::backend::mem_handle const * ph) {return (ph==NULL)?true:memory.insert(std::make_pair((void*)ph, current_arg_)).second; }
      unsigned int get(viennacl::backend::mem_handle const * ph){ return bind(ph)?current_arg_++:memory.at((void*)ph); }
    private:
      unsigned int current_arg_;
      std::map<void*,unsigned int> memory;
    };

    class bind_all_unique : public symbolic_binder{
    public:
      bind_all_unique() : current_arg_(0){ }
      bool bind(viennacl::backend::mem_handle const *) {return true; }
      unsigned int get(viennacl::backend::mem_handle const *){ return current_arg_++; }
    private:
      unsigned int current_arg_;
      std::map<void*,unsigned int> memory;
    };

    enum binding_policy_t{
      BIND_ALL_UNIQUE,
      BIND_TO_HANDLE
    };

    inline tools::shared_ptr<symbolic_binder> make_binder(binding_policy_t policy)
    {
      if(policy==BIND_TO_HANDLE)
        return tools::shared_ptr<symbolic_binder>(new bind_to_handle());
      else
        return tools::shared_ptr<symbolic_binder>(new bind_all_unique());
    }

    class statements_container
    {
    public:
      typedef std::list<scheduler::statement> data_type;
      enum order_type { SEQUENTIAL, INDEPENDENT };

      statements_container(data_type const & data, order_type order) : data_(data), order_(order){ }

      statements_container(scheduler::statement const & s0) : order_(INDEPENDENT)
      {
        data_.push_back(s0);
      }

      statements_container(scheduler::statement const & s0, scheduler::statement const & s1, order_type order) : order_(order)
      {
        data_.push_back(s0);
        data_.push_back(s1);
      }
      std::list<scheduler::statement> const & data() const { return data_; }
      order_type order() const { return order_; }

    private:
      std::list<scheduler::statement> data_;
      order_type order_;
    };



  }

}
#endif
