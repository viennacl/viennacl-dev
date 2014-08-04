#ifndef VIENNACL_DEVICE_SPECIFIC_MAPPED_TYPE_HPP
#define VIENNACL_DEVICE_SPECIFIC_MAPPED_TYPE_HPP

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


/** @file viennacl/generator/mapped_objects.hpp
    @brief Map ViennaCL objects to generator wrappers
*/

#include <string>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/utils.hpp"

namespace viennacl
{

  namespace device_specific
  {

      /** @brief Mapped Object
       *
       * This object populates the symbolic mapping associated with a statement. (root_id, LHS|RHS|PARENT) => mapped_object
       * The tree can then be reconstructed in its symbolic form
       */
      class mapped_object
      {
        private:
           virtual void preprocess(std::string & res) const { }

        protected:
          class MorphBase{ public: virtual std::string operator()(std::string const & i, std::string const & j) const = 0; };

          static void replace_offset(std::string & str, MorphBase const & morph)
          {
              size_t pos = 0;
              while((pos=str.find("#offset", pos))!=std::string::npos)
              {
                  size_t pos_po = str.find('{', pos);
                  size_t pos_comma = str.find(',', pos_po);
                  size_t pos_pe = str.find('}', pos_comma);
                  std::string i = str.substr(pos_po + 1, pos_comma - pos_po - 1);
                  std::string j = str.substr(pos_comma + 1, pos_pe - pos_comma - 1);
                  std::string postprocessed = morph(i, j);
                  str.replace(pos, pos_pe + 1 - pos, postprocessed);
                  pos = pos_pe;
              }
          }

        public:
          struct node_info
          {
              node_info(mapping_type const * _mapping, scheduler::statement const * _statement, vcl_size_t _root_idx) :
                  mapping(_mapping), statement(_statement), root_idx(_root_idx) { }
              mapping_type const * mapping;
              scheduler::statement const * statement;
              vcl_size_t root_idx;
          };

        public:
          mapped_object(std::string const & scalartype, unsigned int id, std::string const & type_key) :
              scalartype_(scalartype), name_("obj"+tools::to_string(id)), type_key_(type_key)
          {
              keywords_["#name"] = name_;
              keywords_["#scalartype"] = scalartype_;
          }

          virtual ~mapped_object(){ }
          virtual std::string & append_kernel_arguments(std::set<std::string> &, std::string & str) const { return str; }
          std::string type_key() const { return type_key_; }

          std::string process(std::string const & in) const
          {
              std::string res(in);
              preprocess(res);
              for(std::map<std::string,std::string>::const_iterator it = keywords_.begin() ; it != keywords_.end() ; ++it)
                  tools::find_and_replace(res, it->first, it->second);
              return res;
          }

          std::string evaluate(std::map<std::string, std::string> const & accessors) const
          {
              if(accessors.find(type_key_)==accessors.end())
                  return name_;
              return process(accessors.at(type_key_));
          }

        protected:
          std::string const name_;
          std::string scalartype_;
          std::string type_key_;
          std::map<std::string, std::string> keywords_;
      };


      /** @brief Binary leaf interface
       *
       *  Some subtrees have to be interpret at leaves when reconstructing the final expression. It is the case of trans(), diag(), prod(), etc...
       *  This interface stores basic infos about the sub-trees
       */
      class binary_leaf
      {
        public:
          binary_leaf(mapped_object::node_info info) : info_(info){ }
          mapping_type const & mapping() const { return *info_.mapping; }
          scheduler::statement const & statement() const { return *info_.statement; }
          vcl_size_t root_idx() const { return info_.root_idx; }
        protected:
          mapped_object::node_info info_;
      };

      /** @brief Matrix product
      *
      * Maps prod(matrix_expression, matrix_expression)
      */
      class mapped_matrix_product : public mapped_object, public binary_leaf
      {
      public:
          mapped_matrix_product(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_product"), binary_leaf(info) { }
      };

      /** @brief Reduction
       *
       * Base class for mapping a reduction
       */
      class mapped_reduction : public mapped_object, public binary_leaf
      {
      public:
          mapped_reduction(std::string const & scalartype, unsigned int id, node_info info, std::string const & type_key) : mapped_object(scalartype, id, type_key), binary_leaf(info){ }
          vcl_size_t root_idx() const { return info_.root_idx; }
          scheduler::statement const & statement() const { return *info_.statement; }
          scheduler::statement_node root_node() const { return statement().array()[root_idx()]; }
      };

      /** @brief Scalar reduction
       *
       * Maps a scalar reduction (max, min, argmax, inner_prod, etc..)
       */
      class mapped_scalar_reduction : public mapped_reduction
      {
      public:
          mapped_scalar_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info, "scalar_reduction"){ }
      };

      /** @brief Vector reduction
       *
       * Maps a row-wise reduction (max, min, argmax, matrix-vector product, etc..)
       */
      class mapped_row_wise_reduction : public mapped_reduction
      {
        public:
          mapped_row_wise_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info, "row_wise_reduction") { }
      };

      /** @brief Host scalar
        *
        * Maps a host scalar (passed by value)
        */
      class mapped_host_scalar : public mapped_object
      {
        public:
          mapped_host_scalar(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "host_scalar"){ }

          std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str) const
          {
            if(already_generated.insert(name_).second)
              str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }
      };

      /** @brief Handle
       *
       * Maps an object passed by pointer
       */
      class mapped_handle : public mapped_object
      {
        private:
          virtual void append_optional_arguments(std::string &) const = 0;

        public:
          mapped_handle(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_object(scalartype, id, type_key){ }

          std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str) const
          {
            if(already_generated.insert(name_).second)
            {
              str += generate_pointer_kernel_argument("__global", scalartype_, name_);
              append_optional_arguments(str);
            }
            return str;
          }
      };


      /** @brief Scalar
        *
        * Maps a scalar passed by pointer
        */
      class mapped_scalar : public mapped_handle
      {
        private:
          void append_optional_arguments(std::string &) const{ }
        public:
          mapped_scalar(std::string const & scalartype, unsigned int id) : mapped_handle(scalartype, id, "scalar") { }
      };

      /** @brief Buffered
        *
        * Maps a buffered object (vector, matrix)
        */
      class mapped_buffer : public mapped_handle
      {
        public:
          mapped_buffer(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_handle(scalartype, id, type_key){ }
      };

      /** @brief Vector
        *
        * Maps a vector
        */
      class mapped_vector : public mapped_buffer
      {
          void append_optional_arguments(std::string & str) const
          {
            str += generate_value_kernel_argument("unsigned int", start_name_);
            str += generate_value_kernel_argument("unsigned int", stride_name_);
          }

        public:
          mapped_vector(std::string const & scalartype, unsigned int id) : mapped_buffer(scalartype, id, "vector")
          {
              start_name_ = name_ + "start";
              stride_name_ = name_ + "stride";

              keywords_["#start"] = start_name_;
              keywords_["#stride"] = stride_name_;
          }

        private:
          std::string start_name_;
          std::string stride_name_;
      };

      /** @brief Matrix
        *
        * Maps a matrix
        */
      class mapped_matrix : public mapped_buffer
      {
        private:
          void append_optional_arguments(std::string & str) const
          {
            str += generate_value_kernel_argument("unsigned int", ld_name_);
            str += generate_value_kernel_argument("unsigned int", start1_name_);
            str += generate_value_kernel_argument("unsigned int", stride1_name_);
            str += generate_value_kernel_argument("unsigned int", start2_name_);
            str += generate_value_kernel_argument("unsigned int", stride2_name_);
          }

          void preprocess(std::string & str) const
          {
            class Morph : public MorphBase
            {
            public:
                Morph(bool row_major) : row_major_(row_major){ }
                std::string operator()(std::string const & i, std::string const & j) const
                {
                    std::string ii = i;
                    std::string jj = j;
                    if(row_major_)
                        std::swap(ii, jj);
                    return "(" + ii + ") + (" + jj + ") * #ld"; }
            private:
                bool row_major_;
            };
            replace_offset(str, Morph(row_major_));
          }

        public:

          mapped_matrix(std::string const & scalartype, unsigned int id, bool row_major) : mapped_buffer(scalartype, id, "matrix"), row_major_(row_major)
          {
              start1_name_ = name_ + "start1";
              start2_name_ = name_ + "start2";
              stride1_name_ = name_ + "stride1";
              stride2_name_ = name_ + "stride2";
              ld_name_ = name_ + "ld";

              keywords_["#ld"] = ld_name_;
              keywords_["#start1"] = start1_name_;
              keywords_["#start2"] = start2_name_;
              keywords_["#stride1"] = stride1_name_;
              keywords_["#stride2"] = stride2_name_;
          }

        private:
          std::string start1_name_;
          std::string start2_name_;

          std::string stride1_name_;
          std::string stride2_name_;

          mutable std::string ld_name_;
          bool row_major_;
      };

      /** @brief Vector diag
       *
       *  Maps a diag(vector_expression) node into a diagonal matrix
       */
      class mapped_vector_diag : public mapped_object, public binary_leaf
      {
      private:
         void preprocess(std::string & res) const
         {
             class Morph : public MorphBase
             {
                 std::string * new_i_;
                 std::string * new_j_;
                 std::string const & rhs_;
             public:
                 Morph(std::string * new_i, std::string * new_j, std::string const & rhs) : new_i_(new_i), new_j_(new_j), rhs_(rhs){ }
                 std::string operator()(std::string const & i, std::string const & j) const
                 {
                     *new_i_ = i + "+ ((" + rhs_ + "<0)?" + rhs_ + ":0)";
                     *new_j_ = j + "- ((" + rhs_ + ">0)?" + rhs_ + ":0)";
                     return "#offset{min("+i+","+j+")}";
                 }
             };

             std::map<std::string, std::string> accessors;
             std::string rhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, RHS_NODE_TYPE);
             std::string new_i, new_j;
             std::string tmp = res;
             replace_offset(tmp, Morph(&new_i, &new_j, rhs));
             accessors["vector"] = tmp;
             std::string lhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, LHS_NODE_TYPE);

             tmp = "((" + new_i + ")!=(" + new_j + "))?0:" + lhs;
         }

      public:
        mapped_vector_diag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix"), binary_leaf(info){ }
      };

      class writable_matrix_modifier : public mapped_object, public binary_leaf
      {
          virtual void replace_offset_impl(std::string & tmp, std::map<std::string, std::string> const & accessors) const = 0;
      protected:
          writable_matrix_modifier(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix"), binary_leaf(info){ }

          void preprocess(std::string & res) const
          {
              std::map<std::string, std::string> accessors;
              std::string tmp = res;
              replace_offset_impl(tmp, accessors);
              accessors["matrix"] = tmp;
              res = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, LHS_NODE_TYPE);
          }

      };

      /** @brief Trans
       *
       *  Maps trans(matrix_expression) into the transposed of matrix_expression
       */
      class mapped_trans: public writable_matrix_modifier
      {
          virtual void replace_offset_impl(std::string & tmp, std::map<std::string, std::string> const &) const
          {
              struct Morph : public MorphBase
              {
                  std::string operator()(std::string const & i, std::string const & j) const
                  { return "#offset{" + j + "," + i + "}"; }
              };
              replace_offset(tmp, Morph());
          }
      public:
        mapped_trans(std::string const & scalartype, unsigned int id, node_info info) : writable_matrix_modifier(scalartype, id, info){ }
      };


      /** @brief Matrix row
       *
       *  Maps row(matrix_expression, scalar_expression) into the scalar_expression's row of matrix_expression
       */
      class mapped_matrix_row : public writable_matrix_modifier
      {
      private:
          virtual void replace_offset_impl(std::string & tmp, std::map<std::string, std::string> const & accessors) const
          {
              class Morph : public MorphBase
              {
              public:
                  Morph(std::string const & idx) : idx_(idx) { }

                  std::string operator()(std::string const & , std::string const & j)  const
                  { return "#offset{" + idx_ + "," + j + "}"; }
              private:
                  std::string const & idx_;
              };
              std::string idx = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, RHS_NODE_TYPE);
              replace_offset(tmp, Morph(idx));
          }
      public:
        mapped_matrix_row(std::string const & scalartype, unsigned int id, node_info info) : writable_matrix_modifier(scalartype, id, info){ }
      };

      /** @brief Matrix column
       *
       *  Maps column(matrix_expression, scalar_expression) into the scalar_expression's column of matrix_expression
       */
      class mapped_matrix_column : public writable_matrix_modifier
      {
      private:
          virtual void replace_offset_impl(std::string & tmp, std::map<std::string, std::string> const & accessors) const
          {
              class Morph : public MorphBase
              {
              public:
                  Morph(std::string const & idx) : idx_(idx) { }

                  std::string operator()(std::string const & i , std::string const & )  const
                  { return "#offset{" + i + "," + idx_ + "}"; }
              private:
                  std::string const & idx_;
              };
              std::string idx = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, RHS_NODE_TYPE);
              replace_offset(tmp, Morph(idx));
          }
      public:
        mapped_matrix_column(std::string const & scalartype, unsigned int id, node_info info) : writable_matrix_modifier(scalartype, id, info){ }
      };

      /** @brief Matrix diag
       *
       *  Maps a diag(matrix_expression) node into the vector of its diagonal elements
       */
      class mapped_matrix_diag : public writable_matrix_modifier
      {
      private:
         virtual void replace_offset_impl(std::string & tmp, std::map<std::string, std::string> const & accessors) const
         {
             class Morph : public MorphBase
             {
             public:
                Morph(std::string const & rhs) : rhs_(rhs){ }

                std::string operator()(std::string const & i, std::string const & j) const
                {
                    std::string new_i = i + "- ((" + rhs_ + "<0)?" + rhs_ + ":0)";
                    std::string new_j = i + "+ ((" + rhs_ + ">0)?" + rhs_ + ":0)";
                    return "#offset{" + new_i + "," + new_j + "}";
                }
             private:
                std::string const & rhs_;
             };

             std::string rhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, accessors, *info_.mapping, RHS_NODE_TYPE);
             replace_offset(tmp, Morph(rhs));
         }
      public:
        mapped_matrix_diag(std::string const & scalartype, unsigned int id, node_info info) : writable_matrix_modifier(scalartype, id, info){ }
      };

      /** @brief Implicit vector
        *
        * Maps an implicit vector
        */
      class mapped_implicit_vector : public mapped_object
      {
        public:
          mapped_implicit_vector(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "implicit_vector"){ }

          std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str) const
          {
            str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }
      };

      /** @brief Implicit matrix
        *
        * Maps an implicit matrix
        */
      class mapped_implicit_matrix : public mapped_object
      {
        public:
          mapped_implicit_matrix(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "implicit_matrix"){ }

          std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str) const
          {
            str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }
      };

    }

}
#endif
