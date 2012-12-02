#ifndef VIENNACL_GENERATOR_CUSTOM_OPERATION_HPP
#define VIENNACL_GENERATOR_CUSTOM_OPERATION_HPP

/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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

/** @file custom_operation.hpp
 *  @brief User Interface for making custom operations. Experimental.
 *
 *  Generator code contributed by Philippe Tillet
 */


#include <vector>
#include <set>
#include <algorithm>

#include "viennacl/generator/get_kernels_infos.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/generator/result_of.hpp"
#include "viennacl/generator/meta_tools/utils.hpp"
#include "viennacl/generator/tweaking.hpp"

namespace viennacl
{
  namespace generator
  {

    /**
     @brief Class to check if a particular symbolic type holds double or not.
     */
    namespace result_of
    {
      template<class T>
      struct is_double_type
      {
        enum { value = are_same_type<double, typename T::ScalarType>::value };
      };

      template<>
      struct is_double_type<NullType>
      {
        enum { value = 0 };
      };
    }




    /** @brief A class for making a custom operation */
    class custom_operation
    {

      private:
        template<class T>
        struct CHECK_OPERATIONS_STRUCTURE
        {
          private:
            template<class U>
            struct is_pure_product_leaf
            {
                enum { value = result_of::is_product_leaf<U>::value && !result_of::is_arithmetic_compound<U>::value};
            };

          public:
            typedef typename tree_utils::extract_if<T,is_pure_product_leaf>::Result Products;
            static const bool is_inplace_product = tree_utils::count_if_type<Products,typename T::LHS>::value;
            static const int n_nested_products = tree_utils::count_if<Products,is_pure_product_leaf>::value - typelist_utils::length<Products>::value;

            static void execute()
            {
                VIENNACL_STATIC_ASSERT(is_inplace_product == false,InplaceProductsForbidden);
                VIENNACL_STATIC_ASSERT(n_nested_products==0,NestedProductsForbidden);
            }
        };




      public :

        /** @brief CTor for 1 expression
        *
        * @param operation_name the code for this expression will be stored in the program provided by this name
        */
        template<class T0>
        custom_operation ( T0 const & , std::string const & operation_name) : program_name_(operation_name)
        {

            typedef typename typelist_utils::make_typelist<T0>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        /** @brief CTor for 2 expressions
        *
        * @param operation_name the code for this expression will be stored in the program provided by this name
        */
        template<class T0,class T1>
        custom_operation ( T0 const & , T1 const & , std::string const & operation_name) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        template<class T0,class T1, class T2>
        custom_operation ( T0 const &, T1 const &, T2 const &, std::string const & operation_name) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1,T2>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        template<class T0,class T1, class T2, class T3>
        custom_operation ( T0 const &, T1 const &, T2 const &, T3 const &, std::string const & operation_name ) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1,T2,T3>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        template<class T0,class T1, class T2, class T3, class T4>
        custom_operation ( T0 const & , T1 const &, T2 const &, T3 const & , T4 const &, std::string const & operation_name ) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1,T2,T3,T4>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        template<class T0,class T1, class T2, class T3, class T4, class T5>
        custom_operation ( T0 const & expr0, T1 const & expr1, T2 const & expr2, T3 const & exp3, T4 const &, T5 const &, std::string const & operation_name ) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1,T2,T3,T4,T5>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        template<class T0,class T1, class T2, class T3, class T4, class T5, class T6>
        custom_operation ( T0 const &, T1 const & , T2 const &, T3 const &, T4 const &, T5 const &, T6 const &, std::string const & operation_name ) : program_name_(operation_name)
        {
            typedef typename typelist_utils::make_typelist<T0,T1,T2,T3,T4,T5,T6>::Result Expressions;
            typedef typename get_operations_from_expressions<Expressions>::Result Operations;
            typelist_utils::ForEach<Operations,CHECK_OPERATIONS_STRUCTURE>::execute();
            viennacl::generator::program_infos<Expressions>::fill(operation_name, sources_,runtime_wrappers_);
            bool has_double = static_cast<bool>(viennacl::generator::tree_utils::count_if<Operations,result_of::is_double_type>::value);
            create_program (has_double);
        }

        /** @brief Returns the list of the kernels involved in the operation
                   in the form of a std::map<std::string,std::string> ( key = kernel_name, value=sources
        */
        std::map<std::string,std::string> const & kernels_sources() const
        {
          return sources_;
        }

        /** @brief Return a string containing the generated source code */
        std::string kernels_source_code() const
        {
          std::string res;
          for (std::map<std::string,std::string>::const_iterator it  = sources_.begin();
                                                                   it != sources_.end();
                                                                 ++it)
          {
            res += it->second + "\n";
          }

          return res;
        }

        /** @brief Returns the name of the program in which the operation is stored */
        std::string const & program_name() const  { return program_name_; }


        /** @brief Convenience for enqueuing the custom operation
            @param t0 first kernel parameter.
        */
        template<class T0>
        custom_operation & operator() ( T0 const & t0)
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          add_operation_arguments();
          return *this;
        }


        /** @brief Convenience for enqueuing the custom operation
            @param t0 first kernel parameter.
            @param t1 first kernel parameter.
        */
        template<class T0, class T1>
        custom_operation & operator() ( T0 const & t0, T1 const & t1 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template<class T0, class T1, class T2>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template<class T0, class T1, class T2, class T3>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template<class T0, class T1, class T2, class T3, class T4>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template<class T0, class T1, class T2, class T3, class T4, class T5>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5, T6 const & t6)
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5, T6 const & t6, T7 const & t7 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9 )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10 >
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11 >
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12 >
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13 >
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14 >
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14,
                  class T15>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14,
                                        T15 const & t15
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          user_args_.insert( std::make_pair(15, viennacl::any(const_cast<T15*>(&t15))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14,
                  class T15, class T16>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14,
                                        T15 const & t15, T16 const & t16
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          user_args_.insert( std::make_pair(15, viennacl::any(const_cast<T15*>(&t15))) );
          user_args_.insert( std::make_pair(16, viennacl::any(const_cast<T16*>(&t16))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14,
                  class T15, class T16, class T17>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14,
                                        T15 const & t15, T16 const & t16, T17 const & t17
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          user_args_.insert( std::make_pair(15, viennacl::any(const_cast<T15*>(&t15))) );
          user_args_.insert( std::make_pair(16, viennacl::any(const_cast<T16*>(&t16))) );
          user_args_.insert( std::make_pair(17, viennacl::any(const_cast<T17*>(&t17))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14,
                  class T15, class T16, class T17, class T18>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14,
                                        T15 const & t15, T16 const & t16, T17 const & t17, T18 const & t18
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          user_args_.insert( std::make_pair(15, viennacl::any(const_cast<T15*>(&t15))) );
          user_args_.insert( std::make_pair(16, viennacl::any(const_cast<T16*>(&t16))) );
          user_args_.insert( std::make_pair(17, viennacl::any(const_cast<T17*>(&t17))) );
          user_args_.insert( std::make_pair(18, viennacl::any(const_cast<T18*>(&t18))) );
          add_operation_arguments();
          return *this;
        }

        /** @brief Convenience for enqueuing the custom operation */
        template <class T0, class T1, class T2, class T3, class T4,
                  class T5, class T6, class T7, class T8, class T9,
                  class T10, class T11, class T12, class T13, class T14,
                  class T15, class T16, class T17, class T18, class T19>
        custom_operation & operator() ( T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                                        T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9,
                                        T10 const & t10, T11 const & t11, T12 const & t12, T13 const & t13, T14 const & t14,
                                        T15 const & t15, T16 const & t16, T17 const & t17, T18 const & t18, T19 const & t19
                                      )
        {
          user_args_.clear();
          user_args_.insert( std::make_pair(0, viennacl::any(const_cast<T0*>(&t0))) );
          user_args_.insert( std::make_pair(1, viennacl::any(const_cast<T1*>(&t1))) );
          user_args_.insert( std::make_pair(2, viennacl::any(const_cast<T2*>(&t2))) );
          user_args_.insert( std::make_pair(3, viennacl::any(const_cast<T3*>(&t3))) );
          user_args_.insert( std::make_pair(4, viennacl::any(const_cast<T4*>(&t4))) );
          user_args_.insert( std::make_pair(5, viennacl::any(const_cast<T5*>(&t5))) );
          user_args_.insert( std::make_pair(6, viennacl::any(const_cast<T6*>(&t6))) );
          user_args_.insert( std::make_pair(7, viennacl::any(const_cast<T7*>(&t7))) );
          user_args_.insert( std::make_pair(8, viennacl::any(const_cast<T8*>(&t8))) );
          user_args_.insert( std::make_pair(9, viennacl::any(const_cast<T9*>(&t9))) );
          user_args_.insert( std::make_pair(10, viennacl::any(const_cast<T10*>(&t10))) );
          user_args_.insert( std::make_pair(11, viennacl::any(const_cast<T11*>(&t11))) );
          user_args_.insert( std::make_pair(12, viennacl::any(const_cast<T12*>(&t12))) );
          user_args_.insert( std::make_pair(13, viennacl::any(const_cast<T13*>(&t13))) );
          user_args_.insert( std::make_pair(14, viennacl::any(const_cast<T14*>(&t14))) );
          user_args_.insert( std::make_pair(15, viennacl::any(const_cast<T15*>(&t15))) );
          user_args_.insert( std::make_pair(16, viennacl::any(const_cast<T16*>(&t16))) );
          user_args_.insert( std::make_pair(17, viennacl::any(const_cast<T17*>(&t17))) );
          user_args_.insert( std::make_pair(18, viennacl::any(const_cast<T18*>(&t18))) );
          user_args_.insert( std::make_pair(19, viennacl::any(const_cast<T19*>(&t19))) );
          add_operation_arguments();
          return *this;
        }


        private:


        void create_program(bool has_double)
        {
          std::string kernels_string;
          for (std::map<std::string,std::string>::iterator it  = sources_.begin();
                                                             it != sources_.end();
                                                           ++it )
          {
            kernels_string += it->second + "\n";
          }
          
          if(has_double)
            kernels_string = viennacl::tools::make_double_kernel(kernels_string,viennacl::ocl::current_device().double_support_extension());

#ifdef VIENNACL_DEBUG_CUSTOM_OPERATION
          std::cout << kernels_string << std::endl;
#endif

          viennacl::ocl::program & program = viennacl::ocl::current_context().add_program(kernels_string, program_name_);

          for (std::map<std::string,std::string>::iterator it  = sources_.begin();
                                                             it != sources_.end();
                                                           ++it)
          {
            program.add_kernel(it->first);
          }
        }


        void add_operation_arguments()
        {
          for (generator::runtime_wrappers_t::iterator it  = runtime_wrappers_.begin();
                                                       it != runtime_wrappers_.end();
                                                     ++it)
          {
            std::string const & kernel_name = it->first;
            viennacl::ocl::kernel& current_kernel = viennacl::ocl::current_context().get_program(program_name_).get_kernel(kernel_name);
            const unsigned int arg_pos = it->second.first;
            generator::result_of::runtime_wrapper * current_arg = it->second.second.get();
        #ifdef VIENNACL_DEBUG_CUSTOM_OPERATION
            std::cout << "Enqueuing : Kernel " << kernel_name << " Argument : " << current_arg->name() << " | Pos : " << arg_pos << std::endl;
        #endif

            current_arg->enqueue(arg_pos,current_kernel,user_args_,temporaries_);

          }
        }

      private :
        std::map<unsigned int, viennacl::any> user_args_;
        std::string program_name_;
        std::vector<viennacl::ocl::local_mem> lmem_;
        std::map<std::string,std::string> sources_;
        viennacl::generator::runtime_wrappers_t runtime_wrappers_;
        std::map<std::string, viennacl::ocl::handle<cl_mem> > temporaries_;
    };


    inline void enqueue_custom_op(viennacl::generator::custom_operation & op, viennacl::ocl::command_queue const & /*queue*/)
    {
      for(std::map<std::string,std::string>::const_iterator it = op.kernels_sources().begin(); it != op.kernels_sources().end() ; ++it)
      {
        std::string current_kernel_name = it->first;
        #ifdef VIENNACL_DEBUG_CUSTOM_OPERATION
        std::cout << "Enqueueing " << current_kernel_name << std::endl;
        #endif
        enqueue(viennacl::ocl::current_context().get_program(op.program_name()).get_kernel(current_kernel_name));
      }
    }

  }
}

#endif

