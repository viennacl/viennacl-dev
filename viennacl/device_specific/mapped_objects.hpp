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
        friend class fetchable;
        friend class writable;

        public:
          struct node_info
          {
              node_info(mapping_type const * _mapping, scheduler::statement const * _statement, vcl_size_t _root_idx) :
                  mapping(_mapping), statement(_statement), root_idx(_root_idx) { }
              mapping_type const * mapping;
              scheduler::statement const * statement;
              vcl_size_t root_idx;
          };

          mapped_object(std::string const & scalartype, unsigned int id) : scalartype_(scalartype), name_("obj"+tools::to_string(id)) {}

          virtual ~mapped_object(){ }

          std::string const & name() const { return name_; }

          std::string const & scalartype() const { return scalartype_; }
          void access_name(std::string const & str) { access_name_ = str; }
          std::string const & access_name() const { return access_name_; }

          virtual std::string & append_kernel_arguments(std::set<std::string> &, std::string & str) const
          { return str; }

          virtual std::string evaluate(index_tuple const & index, unsigned int vector_element) const = 0;

        protected:
          std::string const name_;
          std::string scalartype_;
          std::string access_name_;
      };

      /** @brief Fetchable interface
       *
       *  This interface indicates that an object is fetchable which is the case when it was passed by pointer
       */
      class fetchable
      {
      private:
         virtual std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const = 0;

      protected:
          unsigned int effective_simd_width(unsigned int in) { return force_simd_width_?force_simd_width_:in; }

      public:
        fetchable(mapped_object * obj, unsigned int force_simd_width): obj_(obj), force_simd_width_(force_simd_width){ }

        void fetch(unsigned int simd_width, std::string const & suffix, index_tuple const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream)
        {
          simd_width = effective_simd_width(simd_width);
          obj_->access_name_ = obj_->name_ + suffix;
          if(fetched.insert(obj_->access_name_).second)
            stream << utils::simd_scalartype(obj_->scalartype_, simd_width) << " " << obj_->access_name_ << " = " << fetch_impl(simd_width, index) << ';' << std::endl;
        }
      protected:
        mapped_object * obj_;
        unsigned int force_simd_width_;
      };

      /** @brief Writable interface
       *
       *  This interface indicates that an object is an lvalue. It can be write back to some location in the device's memory
       */
      class writable : public fetchable
      {
      public:
        writable(mapped_object * obj, unsigned int force_simd_width): fetchable(obj, force_simd_width){ }

        virtual void write(unsigned int simd_width, utils::kernel_generation_stream & stream, index_tuple const & index, std::string const & value) const = 0;

        void write_back(unsigned int simd_width, index_tuple const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream)
        {
          simd_width = effective_simd_width(simd_width);
          if(fetched.find(obj_->access_name_)!=fetched.end())
            write(simd_width, stream, index, obj_->access_name_);
        }
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
          mapped_matrix_product(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id), binary_leaf(info){ }
          virtual std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return access_name_ ; }
      };

      /** @brief Reduction
       *
       * Base class for mapping a reduction
       */
      class mapped_reduction : public mapped_object, public binary_leaf
      {
      public:
          mapped_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id), binary_leaf(info){ }
          vcl_size_t root_idx() const { return info_.root_idx; }
          scheduler::statement const & statement() const { return *info_.statement; }
          scheduler::statement_node root_node() const { return statement().array()[root_idx()]; }
          virtual std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return access_name_ ; }
      };

      /** @brief Scalar reduction
       *
       * Maps a scalar reduction (max, min, argmax, inner_prod, etc..)
       */
      class mapped_scalar_reduction : public mapped_reduction
      {
        public:
          mapped_scalar_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info){ }
      };

      /** @brief Vector reduction
       *
       * Maps a row-wise reduction (max, min, argmax, matrix-vector product, etc..)
       */
      class mapped_row_wise_reduction : public mapped_reduction
      {

        public:
          mapped_row_wise_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info){ }
      };

      /** @brief Host scalar
        *
        * Maps a host scalar (passed by value)
        */
      class mapped_host_scalar : public mapped_object
      {
        private:
          std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const { return name_; }

        public:
          mapped_host_scalar(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id){ }

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
      class mapped_handle : public mapped_object, public writable
      {
          friend class writable_matrix_modifier;

        private:
          virtual void append_optional_arguments(std::string &) const = 0;

        public:
          mapped_handle(std::string const & scalartype, unsigned int id, unsigned int force_simd_width) : mapped_object(scalartype, id), writable(this, force_simd_width) { }

          std::string const & name() const
          { return name_; }

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

          std::string fetch_impl(unsigned int, index_tuple const &) const { return "*" + name_; }

          std::string evaluate(index_tuple const & , unsigned int) const { return access_name_; }

          void write(unsigned int simd_width, utils::kernel_generation_stream & stream, index_tuple const & index, std::string const & value) const
          { stream << "*" << name_ << "=" << value << ";" << std::endl; }


        public:
          mapped_scalar(std::string const & scalartype, unsigned int id) : mapped_handle(scalartype, id, 1){ }
      };

      /** @brief Buffered
        *
        * Maps a buffered object (vector, matrix)
        */
      class mapped_buffer : public mapped_handle
      {
        private:
          virtual std::string constant_offset() const = 0;
          virtual std::string element_offset(index_tuple const & index) const = 0;

        public:
          mapped_buffer(std::string const & scalartype, unsigned int id) : mapped_handle(scalartype, id, 0){ }

          std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
          {
              if(simd_width==1)
                return name_  + '[' + constant_offset() + "+" + element_offset(index) + ']';
              else
                return "vload" + tools::to_string(simd_width) + "(" + element_offset(index) + "," + name_ + "+" + constant_offset() + ")";
          }

          void write(unsigned int simd_width, utils::kernel_generation_stream & stream, index_tuple const & index, std::string const & value) const
          {
            if(simd_width==1)
              stream << name_  << "[" << constant_offset() << "+" << element_offset(index) << "] = " << value << ";" << std::endl;
            else
              stream << "vstore" << simd_width << "(" << value << "," << element_offset(index) << "," << name_ + "+" + constant_offset() << ");" << std::endl;
          }

          std::string evaluate(index_tuple const & /*index*/, unsigned int vector_element) const
          {
            if(vector_element==0)
                return access_name_;
            else
                return access_name_ + ".s" + tools::to_string(vector_element);
          }
      };

      /** @brief Vector
        *
        * Maps a vector
        */
      class mapped_vector : public mapped_buffer
      {
          std::string constant_offset() const
          { return start_name_; }

          std::string element_offset(index_tuple const & index) const
          { return index.i+"*"+stride_name_; }

          void append_optional_arguments(std::string & str) const
          {
            str += generate_value_kernel_argument("unsigned int", start_name_);
            str += generate_value_kernel_argument("unsigned int", stride_name_);
          }
        public:
          mapped_vector(std::string const & scalartype, unsigned int id) : mapped_buffer(scalartype, id)
          {
              start_name_ = name_ + "start";
              stride_name_ = name_ + "stride";
          }
        private:
          std::string start_name_;
          std::string stride_name_;
      };

      /** @brief Vector diag
       *
       *  Maps a diag(vector_expression) node into a diagonal matrix
       */
      class mapped_vector_diag : public mapped_object, public binary_leaf, public fetchable
      {
      private:
        std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
        {
            std::string rhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index, 0, *info_.mapping, RHS_NODE_TYPE);
            std::string new_i = index.i + "+ ((" + rhs + "<0)?" + rhs + ":0)";
            std::string new_j = index.j + "- ((" + rhs + ">0)?" + rhs + ":0)";
            index_tuple new_index("min("+index.i+","+index.j+")", index.bound0);
            std::string lhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, new_index, 0, *info_.mapping, LHS_NODE_TYPE);
            return "((" + new_i + ")!=(" + new_j + "))?0:" + lhs;
        }

      public:
        mapped_vector_diag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id), binary_leaf(info), fetchable(this, 0){ }
        std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return access_name_ ; }
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

          std::string constant_offset() const
          {
              std::string i = start1_name_;
              std::string j = start2_name_;
              if(row_major_)
                std::swap(i,j);
              return "(" + i + ") + (" + j + ") *" + ld_name_;
          }

          std::string element_offset(index_tuple const & index) const
          {
              std::string i = "(" + index.i + ")*" + stride1_name_;
              std::string j = "(" + index.j + ")*" + stride2_name_;
              if(row_major_)
                std::swap(i,j);
              return "(" + i + ") + (" + j + ") *" + ld_name_;
          }

        public:
          mapped_matrix(std::string const & scalartype, unsigned int id, bool row_major) : mapped_buffer(scalartype, id), row_major_(row_major)
          {
              start1_name_ = name_ + "start1";
              start2_name_ = name_ + "start2";
              stride1_name_ = name_ + "stride1";
              stride2_name_ = name_ + "stride2";
              ld_name_ = name_ + "ld";
          }

          bool row_major() const { return row_major_; }
          std::string const & ld() const{ return ld_name_; }

        private:
          std::string start1_name_;
          std::string start2_name_;

          std::string stride1_name_;
          std::string stride2_name_;

          mutable std::string ld_name_;
          bool row_major_;
      };

      class writable_matrix_modifier : public mapped_object, public binary_leaf, public writable
      {
      protected:
          writable_matrix_modifier(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id), binary_leaf(info), writable(this, 0){ }
      public:
          void write(unsigned int simd_width, utils::kernel_generation_stream & stream, index_tuple const & index, std::string const & value) const
          { dynamic_cast<mapped_matrix*>(info_.mapping->at(mapping_key(info_.root_idx, LHS_NODE_TYPE)).get())->write(simd_width, stream, index, value); }

          std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return access_name_ ; }
      };

      /** @brief Trans
       *
       *  Maps trans(matrix_expression) into the transposed of matrix_expression
       */
      class mapped_trans: public writable_matrix_modifier
      {
      private:
        std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
        {
          return tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index_tuple(index.j, index.bound1, index.i, index.bound0), 0, *info_.mapping, LHS_NODE_TYPE);
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
        std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
        {
          std::string idx = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index, 0, *info_.mapping, RHS_NODE_TYPE);
          return tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index_tuple(idx,index.bound0, index.i, index.bound0), 0, *info_.mapping, LHS_NODE_TYPE);
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
        std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
        {
          std::string idx = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index, 0, *info_.mapping, RHS_NODE_TYPE);
          return tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index_tuple(index.i,index.bound0, idx, index.bound1), 0, *info_.mapping, LHS_NODE_TYPE);
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
        std::string fetch_impl(unsigned int simd_width, index_tuple const & index) const
        {
          std::string rhs = tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, index, 0, *info_.mapping, RHS_NODE_TYPE);
          std::string new_i = index.i + "- ((" + rhs + "<0)?" + rhs + ":0)";
          std::string new_j = index.i + "+ ((" + rhs + ">0)?" + rhs + ":0)";
          index_tuple new_index(new_i,index.bound0,new_j ,index.bound0);
          return tree_parsing::evaluate_expression(*info_.statement, info_.root_idx, new_index, 0, *info_.mapping, LHS_NODE_TYPE);
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
          mapped_implicit_vector(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id){ }

          std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str) const
          {
            str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }

          std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return name_ ; }
      };

      /** @brief Implicit matrix
        *
        * Maps an implicit matrix
        */
      class mapped_implicit_matrix : public mapped_object
      {
        public:
          mapped_implicit_matrix(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id){ }

          std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str) const
          {
            str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }

          std::string evaluate(index_tuple const & /*index*/, unsigned int /*vector_element*/) const  { return name_ ; }
      };

    }

}
#endif
