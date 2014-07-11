#ifndef VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_HPP
#define VIENNACL_DEVICE_SPECIFIC_TREE_PARSING_HPP

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


/** @file viennacl/generator/helpers.hpp
    @brief several code generation helpers
*/

#include <set>

#include "CL/cl.h"

#include "viennacl/forwards.h"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/utils.hpp"
#include "viennacl/device_specific/forwards.h"

namespace viennacl{

  namespace device_specific{

    namespace tree_parsing{

      /** @brief base functor class for traversing a statement */
      class traversal_functor
      {
        public:
          void call_before_expansion(scheduler::statement const &, vcl_size_t) const { }
          void call_after_expansion(scheduler::statement const &, vcl_size_t) const { }
      };

      /** @brief Recursively execute a functor on a statement */
      template<class Fun>
      inline void traverse(scheduler::statement const & statement, vcl_size_t root_idx, Fun const & fun, bool inspect)
      {
        scheduler::statement_node const & root_node = statement.array()[root_idx];
        bool recurse = utils::node_leaf(root_node.op)?inspect:true;

        fun.call_before_expansion(statement, root_idx);

        //Lhs:
        if(recurse)
        {
          if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.lhs.node_index, fun, inspect);
          if(root_node.lhs.type_family != scheduler::INVALID_TYPE_FAMILY)
            fun(statement, root_idx, LHS_NODE_TYPE);
        }

        //Self:
        fun(statement, root_idx, PARENT_NODE_TYPE);

        //Rhs:
        if(recurse && root_node.rhs.type_family!=scheduler::INVALID_TYPE_FAMILY)
        {
          if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            traverse(statement, root_node.rhs.node_index, fun, inspect);
          if(root_node.rhs.type_family != scheduler::INVALID_TYPE_FAMILY)
            fun(statement, root_idx, RHS_NODE_TYPE);
        }

        fun.call_after_expansion(statement, root_idx);
      }

      class filter : public traversal_functor
      {
        public:
          typedef bool (*pred_t)(scheduler::statement_node const & node);

          filter(pred_t pred, std::vector<vcl_size_t> & out) : pred_(pred), out_(out){ }

          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, leaf_t) const
          {
             scheduler::statement_node const * root_node = &statement.array()[root_idx];
             if(pred_(*root_node))
               out_.push_back(root_idx);
          }
      private:
          pred_t pred_;
          std::vector<vcl_size_t> & out_;
      };

      /** @brief generate a string from an operation_node_type */
      inline const char * evaluate(scheduler::operation_node_type type){
        using namespace scheduler;
        // unary expression
        switch(type){
          //Function
          case OPERATION_UNARY_ABS_TYPE : return "abs";
          case OPERATION_UNARY_ACOS_TYPE : return "acos";
          case OPERATION_UNARY_ASIN_TYPE : return "asin";
          case OPERATION_UNARY_ATAN_TYPE : return "atan";
          case OPERATION_UNARY_CEIL_TYPE : return "ceil";
          case OPERATION_UNARY_COS_TYPE : return "cos";
          case OPERATION_UNARY_COSH_TYPE : return "cosh";
          case OPERATION_UNARY_EXP_TYPE : return "exp";
          case OPERATION_UNARY_FABS_TYPE : return "fabs";
          case OPERATION_UNARY_FLOOR_TYPE : return "floor";
          case OPERATION_UNARY_LOG_TYPE : return "log";
          case OPERATION_UNARY_LOG10_TYPE : return "log10";
          case OPERATION_UNARY_SIN_TYPE : return "sin";
          case OPERATION_UNARY_SINH_TYPE : return "sinh";
          case OPERATION_UNARY_SQRT_TYPE : return "sqrt";
          case OPERATION_UNARY_TAN_TYPE : return "tan";
          case OPERATION_UNARY_TANH_TYPE : return "tanh";

          case OPERATION_UNARY_CAST_CHAR_TYPE : return "(char)";
          case OPERATION_UNARY_CAST_UCHAR_TYPE : return "(uchar)";
          case OPERATION_UNARY_CAST_SHORT_TYPE : return "(short)";
          case OPERATION_UNARY_CAST_USHORT_TYPE : return "(ushort)";
          case OPERATION_UNARY_CAST_INT_TYPE : return "(int)";
          case OPERATION_UNARY_CAST_UINT_TYPE : return "(uint)";
          case OPERATION_UNARY_CAST_LONG_TYPE : return "(long)";
          case OPERATION_UNARY_CAST_ULONG_TYPE : return "(ulong)";
          case OPERATION_UNARY_CAST_HALF_TYPE : return "(half)";
          case OPERATION_UNARY_CAST_FLOAT_TYPE : return "(float)";
          case OPERATION_UNARY_CAST_DOUBLE_TYPE : return "(double)";

          case OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE : return "argfmax";
          case OPERATION_BINARY_ELEMENT_ARGMAX_TYPE : return "argmax";
          case OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE : return "argfmin";
          case OPERATION_BINARY_ELEMENT_ARGMIN_TYPE : return "argmin";
          case OPERATION_BINARY_ELEMENT_POW_TYPE : return "pow";

          //Arithmetic
          case OPERATION_UNARY_MINUS_TYPE : return "-";
          case OPERATION_BINARY_ASSIGN_TYPE : return "=";
          case OPERATION_BINARY_INPLACE_ADD_TYPE : return "+=";
          case OPERATION_BINARY_INPLACE_SUB_TYPE : return "-=";
          case OPERATION_BINARY_ADD_TYPE : return "+";
          case OPERATION_BINARY_SUB_TYPE : return "-";
          case OPERATION_BINARY_MULT_TYPE : return "*";
          case OPERATION_BINARY_ELEMENT_PROD_TYPE : return "*";
          case OPERATION_BINARY_DIV_TYPE : return "/";
          case OPERATION_BINARY_ELEMENT_DIV_TYPE : return "/";
          case OPERATION_BINARY_ACCESS_TYPE : return "[]";

          //Relational
          case OPERATION_BINARY_ELEMENT_EQ_TYPE : return "isequal";
          case OPERATION_BINARY_ELEMENT_NEQ_TYPE : return "isnotequal";
          case OPERATION_BINARY_ELEMENT_GREATER_TYPE : return "isgreater";
          case OPERATION_BINARY_ELEMENT_GEQ_TYPE : return "isgreaterequal";
          case OPERATION_BINARY_ELEMENT_LESS_TYPE : return "isless";
          case OPERATION_BINARY_ELEMENT_LEQ_TYPE : return "islessequal";

          case OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "fmax";
          case OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "fmin";
          case OPERATION_BINARY_ELEMENT_MAX_TYPE : return "max";
          case OPERATION_BINARY_ELEMENT_MIN_TYPE : return "min";
          //Unary
          case OPERATION_UNARY_TRANS_TYPE : return "trans";

          //Binary
          case OPERATION_BINARY_INNER_PROD_TYPE : return "iprod";
          case OPERATION_BINARY_MAT_MAT_PROD_TYPE : return "mmprod";
          case OPERATION_BINARY_MAT_VEC_PROD_TYPE : return "mvprod";
          case OPERATION_BINARY_VECTOR_DIAG_TYPE : return "vdiag";
          case OPERATION_BINARY_MATRIX_DIAG_TYPE : return "mdiag";
          case OPERATION_BINARY_MATRIX_ROW_TYPE : return "row";
          case OPERATION_BINARY_MATRIX_COLUMN_TYPE : return "col";

          default : throw generator_not_supported_exception("Unsupported operator");
        }
      }

      inline const char * operator_string(scheduler::operation_node_type type)
      {
        using namespace scheduler;
        switch(type){
        case OPERATION_UNARY_CAST_CHAR_TYPE : return "char";
        case OPERATION_UNARY_CAST_UCHAR_TYPE : return "uchar";
        case OPERATION_UNARY_CAST_SHORT_TYPE : return "short";
        case OPERATION_UNARY_CAST_USHORT_TYPE : return "ushort";
        case OPERATION_UNARY_CAST_INT_TYPE : return "int";
        case OPERATION_UNARY_CAST_UINT_TYPE : return "uint";
        case OPERATION_UNARY_CAST_LONG_TYPE : return "long";
        case OPERATION_UNARY_CAST_ULONG_TYPE : return "ulong";
        case OPERATION_UNARY_CAST_HALF_TYPE : return "half";
        case OPERATION_UNARY_CAST_FLOAT_TYPE : return "float";
        case OPERATION_UNARY_CAST_DOUBLE_TYPE : return "double";

        case OPERATION_UNARY_MINUS_TYPE : return "umin";
        case OPERATION_BINARY_ASSIGN_TYPE : return "assign";
        case OPERATION_BINARY_INPLACE_ADD_TYPE : return "ip_add";
        case OPERATION_BINARY_INPLACE_SUB_TYPE : return "ip_sub";
        case OPERATION_BINARY_ADD_TYPE : return "add";
        case OPERATION_BINARY_SUB_TYPE : return "sub";
        case OPERATION_BINARY_MULT_TYPE : return "mult";
        case OPERATION_BINARY_ELEMENT_PROD_TYPE : return "eprod";
        case OPERATION_BINARY_DIV_TYPE : return "div";
        case OPERATION_BINARY_ELEMENT_DIV_TYPE : return "ediv";
        case OPERATION_BINARY_ACCESS_TYPE : return "acc";
          default : return evaluate(type);
        }
      }

      /** @brief functor for generating the expression string from a statement */
      class evaluate_expression_traversal: public tree_parsing::traversal_functor
      {
        private:
          index_tuple index_;
          unsigned int simd_element_;
          std::string & str_;
          mapping_type const & mapping_;

        public:
          evaluate_expression_traversal(index_tuple const & index, unsigned int simd_element, std::string & str, mapping_type const & mapping) : index_(index), simd_element_(simd_element), str_(str), mapping_(mapping){ }

          void call_before_expansion(scheduler::statement const & statement, vcl_size_t root_idx) const
          {
              scheduler::statement_node const & root_node = statement.array()[root_idx];
              if((root_node.op.type_family==scheduler::OPERATION_UNARY_TYPE_FAMILY || utils::elementwise_function(root_node.op))
                  && !utils::node_leaf(root_node.op))
                  str_+=tree_parsing::evaluate(root_node.op.type);
              str_+="(";

          }
          void call_after_expansion(scheduler::statement const & /*statement*/, vcl_size_t /*root_idx*/) const
          {
            str_+=")";
          }

          void operator()(scheduler::statement const & statement, vcl_size_t root_idx, leaf_t leaf) const
          {
            scheduler::statement_node const & root_node = statement.array()[root_idx];
            mapping_type::key_type key = std::make_pair(root_idx, leaf);
            if(leaf==PARENT_NODE_TYPE)
            {
              if(utils::node_leaf(root_node.op))
                str_ += mapping_.at(key)->evaluate(index_, simd_element_);
              else if(utils::elementwise_operator(root_node.op))
                str_ += tree_parsing::evaluate(root_node.op.type);
              else if(root_node.op.type_family!=scheduler::OPERATION_UNARY_TYPE_FAMILY && utils::elementwise_function(root_node.op))
                str_ += ",";
            }
            else
            {
              if(leaf==LHS_NODE_TYPE)
              {
                if(root_node.lhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
                  str_ += mapping_.at(key)->evaluate(index_,simd_element_);
              }

              if(leaf==RHS_NODE_TYPE)
              {
                if(root_node.rhs.type_family!=scheduler::COMPOSITE_OPERATION_FAMILY)
                  str_ += mapping_.at(key)->evaluate(index_,simd_element_);
              }
            }
          }
      };

      inline std::string evaluate_expression(scheduler::statement const & statement, vcl_size_t root_idx, index_tuple const & index,
                                             unsigned int simd_element, mapping_type const & mapping, leaf_t leaf)
      {
        std::string res;
        evaluate_expression_traversal traversal_functor(index, simd_element, res, mapping);
        scheduler::statement_node const & root_node = statement.array()[root_idx];

        if(leaf==RHS_NODE_TYPE)
        {
          if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            tree_parsing::traverse(statement, root_node.rhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else if(leaf==LHS_NODE_TYPE)
        {
          if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            tree_parsing::traverse(statement, root_node.lhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else
          tree_parsing::traverse(statement, root_idx, traversal_functor, false);

        return res;
      }


      /** @brief functor for fetching or writing-back the elements in a statement */
      class read_write_traversal : public tree_parsing::traversal_functor{
        public:
          enum mode_t { FETCH, WRITE_BACK };

          read_write_traversal(mode_t mode, std::string suffix, std::set<std::string> & cache,
                               index_tuple const & index, utils::kernel_generation_stream & stream, mapping_type const & mapping)
            : mode_(mode), suffix_(suffix), cache_(cache), index_(index), stream_(stream), mapping_(mapping){ }

          void operator()(scheduler::statement const & /*statement*/, vcl_size_t root_idx, leaf_t leaf) const {
             mapping_type::const_iterator it = mapping_.find(std::make_pair(root_idx, leaf));
             if(it!=mapping_.end())
             {
               if(mode_==FETCH)
                if(fetchable * p = dynamic_cast<fetchable *>(it->second.get()))
                  p->fetch(suffix_, index_, cache_, stream_);

               if(mode_==WRITE_BACK)
                if(writable * p = dynamic_cast<writable *>(it->second.get()))
                  p->write_back(suffix_, index_, cache_, stream_);
             }
          }
      private:
        mode_t mode_;
        std::string suffix_;
        std::set<std::string> & cache_;
        index_tuple index_;
        utils::kernel_generation_stream & stream_;
        mapping_type const & mapping_;
      };


      inline void read_write(tree_parsing::read_write_traversal::mode_t mode, std::string const & suffix,
                                  std::set<std::string> & cache, scheduler::statement const & statement,vcl_size_t root_idx
                                  ,index_tuple const & index,utils::kernel_generation_stream & stream, mapping_type const & mapping, leaf_t leaf)
      {
        read_write_traversal traversal_functor(mode, suffix, cache, index, stream, mapping);
        scheduler::statement_node const & root_node = statement.array()[root_idx];


        if(leaf==RHS_NODE_TYPE)
        {
          if(root_node.rhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            tree_parsing::traverse(statement, root_node.rhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else if(leaf==LHS_NODE_TYPE)
        {
          if(root_node.lhs.type_family==scheduler::COMPOSITE_OPERATION_FAMILY)
            tree_parsing::traverse(statement, root_node.lhs.node_index, traversal_functor, false);
          else
            traversal_functor(statement, root_idx, leaf);
        }
        else
          tree_parsing::traverse(statement, root_idx, traversal_functor, false);
      }


      class statement_representation_functor : public traversal_functor{
        private:
          static void append_id(char * & ptr, unsigned int val){
            if(val==0)
              *ptr++='0';
            else
              while(val>0)
              {
                  *ptr++='0' + (val % 10);
                  val /= 10;
              }
          }

        public:
          typedef void result_type;

          statement_representation_functor(symbolic_binder & binder, char *& ptr) : binder_(binder), ptr_(ptr){ }

          template<class ScalarType>
          inline result_type operator()(ScalarType const & /*scal*/) const
          {
            *ptr_++='h'; //host
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Scalar mapping */
          template<class ScalarType>
          inline result_type operator()(scalar<ScalarType> const & scal) const
          {
            *ptr_++='s'; //scalar
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(scal)));
          }

          /** @brief Vector mapping */
          template<class ScalarType>
          inline result_type operator()(vector_base<ScalarType> const & vec) const
          {
            *ptr_++='v'; //vector
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(vec)));
          }

          /** @brief Implicit vector mapping */
          template<class ScalarType>
          inline result_type operator()(implicit_vector_base<ScalarType> const & /*vec*/) const
          {
            *ptr_++='i'; //implicit
            *ptr_++='v'; //vector
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          /** @brief Matrix mapping */
          template<class ScalarType>
          inline result_type operator()(matrix_base<ScalarType> const & mat) const
          {
            *ptr_++='m'; //Matrix
            *ptr_++=mat.row_major()?'r':'c';
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
            append_id(ptr_, binder_.get(&traits::handle(mat)));
          }

          /** @brief Implicit matrix mapping */
          template<class ScalarType>
          inline result_type operator()(implicit_matrix_base<ScalarType> const & /*mat*/) const
          {
            *ptr_++='i'; //implicit
            *ptr_++='m'; //matrix
            *ptr_++=utils::first_letter_of_type<ScalarType>::value();
          }

          static inline void append(char*& p, const char * str)
          {
            std::size_t n = std::strlen(str);
            std::memcpy(p, str, n);
            p+=n;
          }

          inline void operator()(scheduler::statement const & statement, vcl_size_t root_idx, leaf_t leaf_t) const
          {
            scheduler::statement_node const & root_node = statement.array()[root_idx];
            if(leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node.lhs, *this);
            else if(root_node.op.type_family==scheduler::OPERATION_BINARY_TYPE_FAMILY && leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != scheduler::COMPOSITE_OPERATION_FAMILY)
              utils::call_on_element(root_node.rhs, *this);
            else if(leaf_t==PARENT_NODE_TYPE)
              append_id(ptr_,root_node.op.type);
          }

        private:
          symbolic_binder & binder_;
          char *& ptr_;
      };

      inline std::string statements_representation(statements_container const & statements, binding_policy_t binding_policy)
      {
          std::vector<char> program_name_vector(256);
          char* program_name = program_name_vector.data();
          if(statements.order()==statements_container::INDEPENDENT)
            *program_name++='i';
          else
            *program_name++='s';
          tools::shared_ptr<symbolic_binder> binder = make_binder(binding_policy);
          for(statements_container::data_type::const_iterator it = statements.data().begin() ; it != statements.data().end() ; ++it)
              tree_parsing::traverse(*it, it->root(), tree_parsing::statement_representation_functor(*binder, program_name),true);
          *program_name='\0';
          return std::string(program_name_vector.data());
      }

    }
  }
}
#endif
