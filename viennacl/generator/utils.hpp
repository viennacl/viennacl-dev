#ifndef VIENNACL_GENERATOR_UTILS_HPP
#define VIENNACL_GENERATOR_UTILS_HPP

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


/** @file viennacl/generator/utils.hpp
    @brief Internal utils
*/

#include <sstream>

#include "viennacl/ocl/forwards.h"

#include "viennacl/traits/size.hpp"

#include "viennacl/scheduler/forwards.h"

namespace viennacl{

  namespace generator{

    namespace utils{

    template<class Fun>
    static typename Fun::result_type call_on_host_scalar(scheduler::lhs_rhs_element element, Fun const & fun){
        assert(element.type_family == scheduler::HOST_SCALAR_TYPE_FAMILY && bool("Must be called on a host scalar"));
        switch(element.type){
        case scheduler::FLOAT_TYPE :
            return fun(element.host_float);
        case scheduler::DOUBLE_TYPE :
            return fun(element.host_double);
        default :
            throw "not implemented";
        }
    }

    template<class Fun>
    static typename Fun::result_type call_on_scalar(scheduler::lhs_rhs_element element, Fun const & fun){
        assert(element.type_family == scheduler::SCALAR_TYPE_FAMILY && bool("Must be called on a scalar"));
        switch(element.type){
        case scheduler::FLOAT_TYPE :
            return fun(*element.scalar_float);
        case scheduler::DOUBLE_TYPE :
            return fun(*element.scalar_double);
        default :
            throw "not implemented";
        }
    }

    template<class Fun>
    static typename Fun::result_type call_on_vector(scheduler::lhs_rhs_element element, Fun const & fun){
        assert(element.type_family == scheduler::VECTOR_TYPE_FAMILY && bool("Must be called on a vector"));
        switch(element.type){
        case scheduler::FLOAT_TYPE :
            return fun(*element.vector_float);
        case scheduler::DOUBLE_TYPE :
            return fun(*element.vector_double);
        default :
            throw "not implemented";
        }
    }

    template<class Fun>
    static typename Fun::result_type call_on_symbolic_vector(scheduler::lhs_rhs_element element, Fun const & fun){
        assert(element.type_family == scheduler::SYMBOLIC_VECTOR_TYPE_FAMILY && bool("Must be called on a symbolic_vector"));
        switch(element.type){
        case scheduler::FLOAT_TYPE :
            return fun(*element.symbolic_vector_float);
        case scheduler::DOUBLE_TYPE :
            return fun(*element.symbolic_vector_double);
        default :
            throw "not implemented";
        }
    }

    template<class Fun>
    static typename Fun::result_type call_on_matrix(scheduler::lhs_rhs_element element, Fun const & fun){
        assert((element.type_family == scheduler::MATRIX_ROW_TYPE_FAMILY || element.type_family == scheduler::MATRIX_COL_TYPE_FAMILY) && bool("Must be called on a matrix"));
        if (element.type_family == scheduler::MATRIX_ROW_TYPE_FAMILY)
        {
            switch(element.type){
            case scheduler::FLOAT_TYPE :
                return fun(*element.matrix_row_float);
            case scheduler::DOUBLE_TYPE :
                return fun(*element.matrix_row_double);
            default :
                throw "not implemented";
            }
        }
        else
        {
            switch(element.type){
            case scheduler::FLOAT_TYPE :
                return fun(*element.matrix_col_float);
            case scheduler::DOUBLE_TYPE :
                return fun(*element.matrix_col_double);
            default :
                throw "not implemented";
            }
        }
    }


    template<class Fun>
    static typename Fun::result_type call_on_symbolic_matrix(scheduler::lhs_rhs_element element, Fun const & fun){
        assert(element.type_family == scheduler::SYMBOLIC_MATRIX_TYPE_FAMILY && bool("Must be called on a matrix_vector"));
        switch(element.type){
        case scheduler::FLOAT_TYPE :
            return fun(*element.symbolic_matrix_float);
        case scheduler::DOUBLE_TYPE :
            return fun(*element.symbolic_matrix_double);
        default :
            throw "not implemented";
        }
    }

      template<class Fun>
      static typename Fun::result_type call_on_element(scheduler::lhs_rhs_element const & element, Fun const & fun){
        switch(element.type_family){
          case scheduler::HOST_SCALAR_TYPE_FAMILY:
            return call_on_host_scalar(element, fun);
          case scheduler::SCALAR_TYPE_FAMILY:
            return call_on_scalar(element, fun);
          case scheduler::VECTOR_TYPE_FAMILY :
            return call_on_vector(element, fun);
          case scheduler::SYMBOLIC_VECTOR_TYPE_FAMILY :
            return call_on_symbolic_vector(element, fun);
          case scheduler::MATRIX_ROW_TYPE_FAMILY:
            return call_on_matrix(element,fun);
          case scheduler::MATRIX_COL_TYPE_FAMILY:
            return call_on_matrix(element,fun);
          case scheduler::SYMBOLIC_MATRIX_TYPE_FAMILY :
            return call_on_symbolic_matrix(element, fun);
          default:
            throw "not implemented";
        }
      }

      struct size_fun{
          typedef std::size_t result_type;
          template<class T>
          result_type operator()(T const &t) const { return viennacl::traits::size(t); }
      };

      struct handle_fun{
          typedef cl_mem result_type;
          template<class T>
          result_type operator()(T const &t) const { return t.handle().opencl_handle(); }
      };

      struct size1_fun{
          typedef std::size_t result_type;
          template<class T>
          result_type operator()(T const &t) const { return viennacl::traits::size1(t); }
      };

      struct size2_fun{
          typedef std::size_t result_type;
          template<class T>
          result_type operator()(T const &t) const { return viennacl::traits::size2(t); }
      };

//      static std::size_t size(scheduler::statement_node_type type, scheduler::lhs_rhs_element element){
//        return call_on_vector(type, element, size_fun());
//      }
      
//      static std::size_t size1(scheduler::statement_node_type type, scheduler::lhs_rhs_element element){
//        return call_on_matrix(type, element, size1_fun());
//      }

//      static std::size_t size2(scheduler::statement_node_type type, scheduler::lhs_rhs_element element){
//        return call_on_matrix(type, element, size2_fun());
//      }

      template<class T, class U>
      struct is_same_type { enum { value = 0 }; };

      template<class T>
      struct is_same_type<T,T> { enum { value = 1 }; };

      template <class T>
      inline std::string to_string ( T const t )
      {
        std::stringstream ss;
        ss << t;
        return ss.str();
      }

      template<class T>
      struct type_to_string;
      template<> struct type_to_string<float> { static const char * value() { return "float"; } };
      template<> struct type_to_string<double> { static const char * value() { return "double"; } };


      template<class T>
      struct first_letter_of_type;
      template<> struct first_letter_of_type<float> { static char value() { return 'f'; } };
      template<> struct first_letter_of_type<double> { static char value() { return 'd'; } };
      template<> struct first_letter_of_type<viennacl::row_major> { static char value() { return 'r'; } };
      template<> struct first_letter_of_type<viennacl::column_major> { static char value() { return 'c'; } };

      class kernel_generation_stream : public std::ostream{
        private:

          class kgenstream : public std::stringbuf{
            public:
              kgenstream(std::ostringstream& oss,unsigned int const & tab_count) : oss_(oss), tab_count_(tab_count){ }
              int sync() {
                for(unsigned int i=0 ; i<tab_count_;++i)
                  oss_ << "    ";
                oss_ << str();
                str("");
                return !oss_;
              }
              ~kgenstream() {  pubsync(); }
            private:
              std::ostream& oss_;
              unsigned int const & tab_count_;
          };

        public:
          kernel_generation_stream() : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0){ }

          std::string str(){ return oss.str(); }

          void inc_tab(){ ++tab_count_; }

          void dec_tab(){ --tab_count_; }

          ~kernel_generation_stream(){ delete rdbuf(); }

        private:
          unsigned int tab_count_;
          std::ostringstream oss;
      };


    }

  }

}
#endif
