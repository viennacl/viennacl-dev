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

namespace viennacl{

  namespace device_specific{

      /** @brief Base class for mapping viennacl datastructure to generator-friendly structures
       */
      class mapped_object{
        protected:
          struct node_info{
              node_info() : mapping(NULL), statement(NULL), root_node(NULL) { }
              mapping_type const * mapping;
              scheduler::statement const * statement;
              scheduler::statement_node const * root_node;
          };
          virtual std::string generate_default(std::pair<std::string, std::string> const & index) const = 0;
        public:
          mapped_object(std::string const & scalartype) : scalartype_(scalartype){          }
          virtual std::string & append_kernel_arguments(std::set<std::string> &, std::string & str) const{ return str; }
          std::string const & scalartype() const { return scalartype_; }
          void access_name(std::string const & str) { access_name_ = str; }
          std::string const & access_name() const { return access_name_; }
          virtual std::string evaluate(std::pair<std::string, std::string> const & index, int) const{
            if(!access_name_.empty())
              return access_name_;
            else
              return generate_default(index);
          }
          virtual ~mapped_object(){ }
        protected:
          std::string access_name_;
          std::string scalartype_;
      };

      /** @brief Base class for mapping binary leaves (inner product-based, matrix vector product-base, matrix-matrix product based...)
       */
      class mapped_binary_leaf : public mapped_object{
        public:
          mapped_binary_leaf(std::string const & scalartype) : mapped_object(scalartype){ }
          mapping_type const & mapping() const { return *info_.mapping; }
          scheduler::statement const & statement() const { return *info_.statement; }
          scheduler::statement_node const & root_node() const { return *info_.root_node; }
          std::string generate_default(std::pair<std::string, std::string> const &) const { return "";}
        protected:
          node_info info_;
      };

      /** @brief Mapping of a matrix product */
      class mapped_matrix_product : public mapped_binary_leaf{
          friend class tree_parsing::map_functor;
        public:
          mapped_matrix_product(std::string const & scalartype) : mapped_binary_leaf(scalartype){ }
      };

      /** @brief Base class for mapping a reduction */
      class mapped_reduction : public mapped_binary_leaf{
        public:
          mapped_reduction(std::string const & scalartype) : mapped_binary_leaf(scalartype){ }
          scheduler::operation_node_type reduction_type() const { return reduction_type_; }
        private:
          scheduler::operation_node_type reduction_type_;
      };

      /** @brief Mapping of a scalar reduction (based on inner product) */
      class mapped_scalar_reduction : public mapped_reduction{
          friend class tree_parsing::map_functor;
        public:
          mapped_scalar_reduction(std::string const & scalartype) : mapped_reduction(scalartype){ }
      };

      /** @brief Mapping of a vector reduction (based on matrix-vector product) */
      class mapped_vector_reduction : public mapped_reduction{
          friend class tree_parsing::map_functor;
        public:
          mapped_vector_reduction(std::string const & scalartype) : mapped_reduction(scalartype){ }
      };

      /** @brief Mapping of a host scalar to a generator class */
      class mapped_host_scalar : public mapped_object{
          friend class tree_parsing::map_functor;
          std::string generate_default(std::pair<std::string, std::string> const &) const{ return name_;  }
        public:
          mapped_host_scalar(std::string const & scalartype) : mapped_object(scalartype){ }
          std::string const & name() { return name_; }
          std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str) const{
            if(already_generated.insert(name_).second)
              str += generate_value_kernel_argument(scalartype_, name_);
            return str;
          }

        private:
          std::string name_;
      };

      /** @brief Base class for datastructures passed by pointer */
      class mapped_handle : public mapped_object{
          virtual std::string offset(std::pair<std::string, std::string> const & index) const = 0;
          virtual void append_optional_arguments(std::string &) const{ }
          std::string generate_default(std::pair<std::string, std::string> const & index) const{ return name_  + '[' + offset(index) + ']'; }
        public:
          mapped_handle(std::string const & scalartype) : mapped_object(scalartype){ }

          std::string const & name() const { return name_; }
          void set_simd_width(unsigned int val) {  simd_width_ = val; }
          unsigned int simd_width() const{ return simd_width_; }

          std::string simd_scalartype() const{
              std::string res = scalartype_;
              if(simd_width_ > 1)
                  res+=utils::to_string(simd_width_);
              return res;
          }

          void fetch(std::pair<std::string, std::string> const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream) {
            std::string new_access_name = name_ + "_private";
            if(fetched.find(name_)==fetched.end()){
              stream << scalartype_;
              if(simd_width_ > 1)
                  stream << simd_width_;
              stream << " " << new_access_name << " = " << generate_default(index) << ';' << std::endl;
              fetched.insert(name_);
            }
            access_name_ = new_access_name;
          }

          void write_back(std::pair<std::string, std::string> const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream) {
            std::string old_access_name = access_name_ ;
            access_name_ = "";
            if(fetched.find(name_)!=fetched.end()){
              stream << generate_default(index) << " = " << old_access_name << ';' << std::endl;
              fetched.erase(name_);
            }
          }

          std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str) const{
            if(already_generated.insert(name_).second){
              std::string vector_scalartype = scalartype_;
              if(simd_width_>1)
                vector_scalartype+=utils::to_string(simd_width_);
              str += generate_pointer_kernel_argument("__global", vector_scalartype, name_);
              append_optional_arguments(str);
            }
            return str;
          }

        protected:
          std::string name_;
          unsigned int simd_width_;
      };

      /** @brief Mapping of a scalar to a generator class */
      class mapped_scalar : public mapped_handle{
          friend class tree_parsing::map_functor;
        private:
          std::string offset(std::pair<std::string, std::string> const &)  const { return "0"; }
        public:
          mapped_scalar(std::string const & scalartype) : mapped_handle(scalartype){ }
      };


      /** @brief Base class for mapping buffer-based objects to a generator class */
      class mapped_buffer : public mapped_handle{
        public:
          mapped_buffer(std::string const & scalartype) : mapped_handle(scalartype){ }
          virtual std::string evaluate(std::pair<std::string, std::string> const & index, int vector_element) const{
            if(vector_element>-1)
              return mapped_object::evaluate(index, vector_element)+".s"+utils::to_string(vector_element);
            return mapped_object::evaluate(index, vector_element);
          }

      };

      /** @brief Mapping of a vector to a generator class */
      class mapped_vector : public mapped_buffer{
          friend class tree_parsing::map_functor;
          std::string offset(std::pair<std::string, std::string> const & index) const {
            if(info_.statement){
              std::string str;
              tree_parsing::generate_all_rhs(*info_.statement, *info_.root_node, index, -1, str, *info_.mapping);
              return str;
            }
            else
              return start_name_+"+"+index.first+"*"+stride_name_;
          }

          void append_optional_arguments(std::string & str) const{
            str += generate_value_kernel_argument("unsigned int", start_name_);
            str += generate_value_kernel_argument("unsigned int", stride_name_);
          }
        public:
          mapped_vector(std::string const & scalartype) : mapped_buffer(scalartype){ }
        private:
          node_info info_;

          std::string start_name_;
          std::string stride_name_;
      };

      /** @brief Mapping of a matrix to a generator class */
      class mapped_matrix : public mapped_buffer{
          friend class tree_parsing::map_functor;
          void append_optional_arguments(std::string & str) const{
            str += generate_value_kernel_argument("unsigned int", ld_name_);
            str += generate_value_kernel_argument("unsigned int", start1_name_);
            str += generate_value_kernel_argument("unsigned int", stride1_name_);
            str += generate_value_kernel_argument("unsigned int", start2_name_);
            str += generate_value_kernel_argument("unsigned int", stride2_name_);
          }
        public:
          mapped_matrix(std::string const & scalartype) : mapped_buffer(scalartype){ }

          bool interpret_as_transposed() const { return interpret_as_transposed_; }

          std::string const & ld() const { return ld_name_; }

          std::string offset(std::pair<std::string, std::string> const & index) const {
            std::string i = index.first;
            std::string j = index.second;
            if(i=="0")
              return  "(" + j + ')' + '*' + ld_name_;
            else if(j=="0")
              return "(" + i + ")";
            else
              return  '(' + i + ')' + "+ (" + j + ')' + '*' + ld_name_;
          }

        private:
          std::string start1_name_;
          std::string start2_name_;

          std::string stride1_name_;
          std::string stride2_name_;

          mutable std::string ld_name_;
          bool interpret_as_transposed_;
      };

      /** @brief Mapping of a implicit vector to a generator class */
      class mapped_implicit_vector : public mapped_object{
          friend class tree_parsing::map_functor;
          std::string value_name_;
          std::string index_name_;
          bool is_value_static_;
        public:
          mapped_implicit_vector(std::string const & scalartype) : mapped_object(scalartype){ }
          std::string generate_default(std::pair<std::string, std::string> const & /*index*/) const{
            return value_name_;
          }
          std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str) const{
            if(!value_name_.empty())
              str += generate_value_kernel_argument(scalartype_, value_name_);
            if(!index_name_.empty())
              str += generate_value_kernel_argument("unsigned int", index_name_);
            return str;
          }
      };

      /** @brief Mapping of a implicit matrix to a generator class */
      class mapped_implicit_matrix : public mapped_object{
          friend class tree_parsing::map_functor;
          std::string value_name_;
          bool is_diag_;
        public:
          mapped_implicit_matrix(std::string const & scalartype) : mapped_object(scalartype){ }
          std::string generate_default(std::pair<std::string, std::string> const & /* index */) const{
            return value_name_;
          }
          std::string & append_kernel_arguments(std::set<std::string> & /*already generated*/, std::string & str) const{
            if(!value_name_.empty())
              str += generate_value_kernel_argument(scalartype_, value_name_);
            return str;
          }
      };

      static std::string evaluate(std::pair<std::string, std::string> const & index, int simd_element, mapped_object const & s){
        return s.evaluate(index, simd_element);
      }

      static void fetch(std::pair<std::string, std::string> const & index, std::set<std::string> & fetched, utils::kernel_generation_stream & stream, mapped_object & s){
        if(mapped_handle * p = dynamic_cast<mapped_handle  *>(&s))
          p->fetch(index, fetched, stream);
      }

      static std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str, mapped_object const & s){
        return s.append_kernel_arguments(already_generated, str);
      }

    }

}
#endif
