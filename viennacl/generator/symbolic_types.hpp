#ifndef VIENNACL_GENERATOR_SYMBOLIC_TYPES_HPP
#define VIENNACL_GENERATOR_SYMBOLIC_TYPES_HPP

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


/** @file viennacl/generator/symbolic_types.hpp
 *  @brief Definition of the symbolic types. Experimental.
 *
 *  Generator code contributed by Philippe Tillet
 */

#include "viennacl/forwards.h"
#include "viennacl/generator/result_of.hpp"
#include "viennacl/generator/meta_tools/utils.hpp"


namespace viennacl
{
  namespace generator
  {

    /**
    * @brief Symbolic constant. Used for elementwise operations.

    * @tparam VAL value of the constant.
    */
    template<long VAL>
    class symbolic_constant
    {
    public:
        static const long value = VAL;
        typedef long ScalarType;
        static std::string name()
        {
            return "symcst"+to_string(value);
        }
    };

    /**
    * @brief Symbolic scalar type. Will be passed by value.
    *
    * @tparam ID The argument ID of the scalar in the generated code
    * @tparam SCALARTYPE The Scalartype of the scalar in the generated code
    */
    template <unsigned int ID, typename SCALARTYPE>
    class cpu_symbolic_scalar
    {
      private:
        typedef cpu_symbolic_scalar<ID,SCALARTYPE> self_type;

      public:
        static const unsigned int Alignment = 1;
        typedef SCALARTYPE ScalarType;
        typedef ScalarType runtime_type;
        enum { id = ID };
        static std::string name()
        {
          std::ostringstream oss;
          oss << "c_s" << ID ;
          return oss.str();
        }
        static std::string kernel_arguments()
        {
          return print_type<SCALARTYPE,1>::value() + " " + name() + "\n";
        }
    };

    /**
    * @brief Symbolic scalar type. Will be passed by pointer.
    *
    * @tparam ID The argument ID of the scalar in the generated code
    * @tparam SCALARTYPE The Scalartype of the scalar in the generated code
    */
    template <unsigned int ID, typename SCALARTYPE>
    class gpu_symbolic_scalar
    {
      private:
        typedef gpu_symbolic_scalar<ID,SCALARTYPE> self_type;
      public:
        static const unsigned int Alignment = 1;
        typedef SCALARTYPE ScalarType;
        typedef viennacl::scalar<ScalarType> runtime_type;
        enum { id = ID };

        static std::string val_name()
        {
            return "val_" + name();
        }

        static std::string declarations()
        {
            return print_type<ScalarType,1>::value() + " " +  val_name() + "= " "*"+name() + ";\n";
        }

        static std::string assignements()
        {
            return "*"+name() + "="  +  val_name() + ";\n";
        }

        static std::string name()
        {
          std::ostringstream oss;
          oss << "g_s" << ID ;
          return oss.str();
        }


        static std::string kernel_arguments()
        {
          return "__global " + print_type<SCALARTYPE*,1>::value() + " " + name() + "\n" ;
        }

        template<typename RHS_TYPE>
        typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                          compound_node<self_type, assign_type, RHS_TYPE > >::type
        operator= ( RHS_TYPE const & ) const
        {
          return compound_node<self_type,assign_type,RHS_TYPE >();
        }

        template<typename RHS_TYPE>
        typename enable_if<result_of::is_scalar_expression<RHS_TYPE>,
                          compound_node<self_type, inplace_scal_mul_type, RHS_TYPE > >::type
        operator*= ( RHS_TYPE const & ) const
        {
          return compound_node<self_type,inplace_scal_mul_type,RHS_TYPE >();
        }

        template<typename RHS_TYPE>
        typename enable_if<result_of::is_scalar_expression<RHS_TYPE>,
                          compound_node<self_type, inplace_scal_div_type, RHS_TYPE > >::type
        operator/= ( RHS_TYPE const & ) const
        {
          return compound_node<self_type,inplace_scal_div_type,RHS_TYPE >();
        }

        template<typename RHS_TYPE>
        typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                          compound_node<self_type, inplace_add_type, RHS_TYPE > >::type
        operator+= ( RHS_TYPE const & ) const
        {
          return compound_node<self_type,inplace_add_type,RHS_TYPE >();
        }

        template<typename RHS_TYPE>
        typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                          compound_node<self_type, inplace_sub_type, RHS_TYPE > >::type
        operator-= ( RHS_TYPE const & ) const
        {
          return compound_node<self_type,inplace_sub_type,RHS_TYPE >();
        }

    };


      /**
      * @brief Symbolic vector type
      *
      * @tparam ID The argument ID of the vector in the generated code
      * @tparam SCALARTYPE The Scalartype of the vector in the generated code
      * @tparam ALIGNMENT The Alignment of the vector in the generated code
      */

      //TODO: Add start and inc...
      template <unsigned int ID, typename SCALARTYPE, unsigned int ALIGNMENT>
      class symbolic_vector
      {
        private:
          typedef symbolic_vector<ID,SCALARTYPE,ALIGNMENT> self_type;

        public:

          typedef SCALARTYPE ScalarType;

          static const unsigned int Alignment = ALIGNMENT;

          typedef viennacl::vector<ScalarType,Alignment> runtime_type;

          static const unsigned int id = ID;

          static std::string gid_val_name(){
              return "gid_val_"+name();
          }

          static std::string declarations(){
              return print_type<ScalarType,Alignment>::value() + " " + gid_val_name() + "=" + name()+"[gid];\n";
          }

          static std::string assignements(){
              return  name()+"[gid]" + " = "  + gid_val_name() + ";\n";
          }

          static std::string name()
          {
            return "v_a" + to_string(Alignment) + "_" + to_string(ID);
          }

          static std::string size2_name()
          {
            return "size_"+name();
          }

          static std::string internal_size2_name()
          {
            return "internal_size_"+name();
          }

          static std::string name_argument()
          {
            return " __global " + print_type<SCALARTYPE*,Alignment>::value() + " " + name();
          }

          static std::string kernel_arguments()
          {
            return " __global " + print_type<SCALARTYPE*,Alignment>::value() + " " + name()
                + ", unsigned int " + size2_name()
                + ", unsigned int " + internal_size2_name() + "\n" ;
          }

          template<typename RHS_TYPE>
          typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, assign_type, RHS_TYPE > >::type
          operator= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,assign_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<result_of::is_scalar_expression<RHS_TYPE>,
                            compound_node<self_type, inplace_scal_mul_type, RHS_TYPE > >::type
          operator*= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_scal_mul_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<result_of::is_scalar_expression<RHS_TYPE>,
                            compound_node<self_type, inplace_scal_div_type, RHS_TYPE > >::type
          operator/= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_scal_div_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, inplace_add_type, RHS_TYPE > >::type
          operator+= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_add_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, inplace_sub_type, RHS_TYPE > >::type
          operator-= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_sub_type,RHS_TYPE >();
          }

          operator compound_node<self_type,assign_type,self_type>()
          {
            return compound_node<self_type,assign_type,self_type>();
          }
      };




      /**
      * @brief Symbolic matrix type
      *
      * @tparam ID The argument ID of the matrix in the generated code
      * @tparam SCALARTYPE The Scalartype of the matrix in the generated code
      * @tparam F The Layout of the matrix in the generated code
      * @tparam ALIGNMENT The Alignment of the matrix in the generated code
      */
      template<unsigned int ID, typename SCALARTYPE, class F, unsigned int ALIGNMENT>
      class symbolic_matrix
      {

          typedef symbolic_matrix<ID, SCALARTYPE, F, ALIGNMENT> self_type;

        public:

          enum { id = ID };
          typedef SCALARTYPE ScalarType;
          typedef F Layout;
          static const unsigned int Alignment = ALIGNMENT;

          typedef viennacl::matrix<ScalarType,F,Alignment> runtime_type;
          static std::string gid_val_name()
          {
              return "gid_val_"+name();
          }

          static std::string declarations()
          {
            return print_type<ScalarType,Alignment>::value() + " " + gid_val_name() + "=" + name()+"[gid];\n";
          }

          static std::string assignements()
          {
            return  name()+"[gid]" + " = " + gid_val_name() + ";\n";
          }

          static std::string name()
          {
            F layout;
            return "m_a_" + viennacl::generator::to_string(layout) + "_"
                          + viennacl::generator::to_string(Alignment) + "_"
                          + viennacl::generator::to_string<long>(id);
          }


          static std::string row_inc_name()
          {
            return name()+"_row_inc";
          }

          static std::string col_inc_name()
          {
            return name()+"_col_inc";
          }

          static std::string row_start_name()
          {
            return name()+"_row_start";
          }

          static std::string col_start_name()
          {
            return name()+"_col_start";
          }

          static std::string size1_name()
          {
            return "size1_" + name();
          }

          static std::string size2_name()
          {
            return "size2_" + name();
          }

          static std::string internal_size1_name()
          {
            return "internal_size1_" + name();
          }

          static std::string internal_size2_name()
          {
            return "internal_size2_" + name();
          }

          static std::string kernel_arguments()
          {
            return " __global " + generator::print_type<SCALARTYPE*,Alignment>::value() + " " + name()
                  + ", unsigned int " + row_start_name()
                  + ", unsigned int " + col_start_name()
                  + ", unsigned int " + row_inc_name()
                  + ", unsigned int " + col_inc_name()
                  + ", unsigned int " + size1_name()
                  + ", unsigned int " + size2_name()
                  + ", unsigned int " + internal_size1_name()
                  + ", unsigned int " + internal_size2_name()
                  + "\n";
          }

          template<typename RHS_TYPE>
          typename enable_if<generator::result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, assign_type, RHS_TYPE > >::type
          operator= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,assign_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<generator::result_of::is_scalar_expression<RHS_TYPE>,
                            compound_node<self_type, inplace_scal_mul_type, RHS_TYPE > >::type
          operator*= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_scal_mul_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<generator::result_of::is_scalar_expression<RHS_TYPE>,
                            compound_node<self_type, inplace_scal_div_type, RHS_TYPE > >::type
          operator/= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_scal_div_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<generator::result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, inplace_add_type, RHS_TYPE > >::type
          operator+= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_add_type,RHS_TYPE >();
          }

          template<typename RHS_TYPE>
          typename enable_if<generator::result_of::is_same_expression_type<self_type,RHS_TYPE>,
                            compound_node<self_type, inplace_sub_type, RHS_TYPE > >::type
          operator-= ( RHS_TYPE const & ) const
          {
            return compound_node<self_type,inplace_sub_type,RHS_TYPE >();
          }

          operator compound_node<self_type,assign_type,self_type>() 
          {
            return compound_node<self_type,assign_type,self_type>();
          }
      };

      /**
      * @brief Binary node class for storing expression trees
      *
      * @tparam LHS_ LHS of the expression
      * @tparam OP_ Operator of the expression
      * @tparam RHS_ RHS of the expression
      */
      template<class LHS_, class OP_, class RHS_>
      class compound_node
      {
        public:
          typedef LHS_  LHS;
          typedef RHS_  RHS;
          typedef OP_   OP;

          typedef typename result_of::expression_type<RHS>::Result IntermediateType;  //Note: Visual Studio does not allow to combine this line with the next one directly.
          typedef typename IntermediateType::ScalarType ScalarType;

          static std::string name()
          {
              return LHS::name() + "_" + OP::name() + "_" + RHS::name();
          }
      };

      template<class LHS_, class RHS_>
      class compound_node<LHS_,inner_prod_type,RHS_>
      {
        public:
          /**
          * @brief Specialization for the inner product
          */
          typedef LHS_ LHS;
          typedef RHS_ RHS;
          typedef inner_prod_type OP;
          typedef typename result_of::expression_type<LHS>::Result IntermediateType;  //Note: Visual Studio does not allow to combine this line with the next one directly.
          typedef typename IntermediateType::ScalarType ScalarType;
          static const unsigned int Alignment = IntermediateType::Alignment;

          enum { id = -2 };

          static std::string kernel_arguments()
          {
              return  "__global " + print_type<ScalarType*,1>::value() + " " + name() + '\n';
          }

          static std::string name()
          {
              return  LHS::name() + "_inprod_" + RHS::name();
          }

          static std::string local_value()
          {
              return "local_"+name();
          }

          static std::string declarations()
          {
              return "__local " + print_type<ScalarType,1>::value() + " " + local_value() + ";\n";
          }

          static std::string scalar_name()
          {
            return name() +"_s";
          }

      };

      /**
      * @brief Specialization for the matrix-vector product.
      */
      template<class LHS_, class RHS_>
      class compound_node<LHS_,prod_type,RHS_>
      {
        private:
          typedef compound_node<LHS_,prod_type,RHS_> self_type;

        public:
          typedef LHS_ LHS;
          typedef RHS_ RHS;

          typedef prod_type OP;
          enum { id = LHS::id };

          typedef typename result_of::expression_type<LHS>::Result IntermediateType;    //Note: Visual Studio does not allow to combine this line with the next one directly.
          typedef typename IntermediateType::ScalarType ScalarType;
          static const unsigned int Alignment = result_of::expression_type<LHS>::Result::Alignment;

          static std::string name()
          {
            return LHS::name() + "_prod_" + RHS::name();
          }
      };


  } // namespace generator
} // namespace viennacl


#endif

