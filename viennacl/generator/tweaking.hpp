#ifndef VIENNACL_GENERATOR_TWEAKING_HPP
#define VIENNACL_GENERATOR_TWEAKING_HPP

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

#include "viennacl/generator/meta_tools/typelist.hpp"
#include "viennacl/generator/symbolic_types.hpp"

/** @file viennacl/generator/tweaking.hpp
 *  @brief Additional operations on expressions. Experimental.
 * 
 *  Generator code contributed by Philippe Tillet
 */

namespace viennacl
{
  namespace generator
  {

    static const symbolic_constant<1>  _1_ = symbolic_constant<1>();
    static const symbolic_constant<2>  _2_ = symbolic_constant<2>();
    static const symbolic_constant<3>  _3_ = symbolic_constant<3>();
    static const symbolic_constant<4>  _4_ = symbolic_constant<4>();
    static const symbolic_constant<5>  _5_ = symbolic_constant<5>();
    static const symbolic_constant<6>  _6_ = symbolic_constant<6>();
    static const symbolic_constant<7>  _7_ = symbolic_constant<7>();
    static const symbolic_constant<8>  _8_ = symbolic_constant<8>();
    static const symbolic_constant<9>  _9_ = symbolic_constant<9>();




    template<class T>
    typename viennacl::enable_if<result_of::is_matrix_expression<T>::value, compound_node<T,prod_type,symbolic_constant<1> > >::type
    sum(T t, symbolic_constant<1>)
    {
        return compound_node<T,prod_type,symbolic_constant<1> >();
    }

    template<class T>
    typename viennacl::enable_if<result_of::is_vector_expression<T>::value, compound_node<T,inner_prod_type,symbolic_constant<1> > >::type
    sum(T t)
    {
        return compound_node<T,inner_prod_type,symbolic_constant<1> >();
    }


    template<class Bound_, class Operations_>
    struct repeater_impl{
        typedef Operations_ Operations;
        typedef Bound_ Bound;
    };

    template<class Bound, class T>
    repeater_impl<Bound, VIENNACL_TYPELIST1(T) > repeat(Bound , T){
        return repeater_impl<Bound, VIENNACL_TYPELIST1(T) > ();
    }

    template<class Bound, class T, class T2>
    repeater_impl<Bound, VIENNACL_TYPELIST2(T,T2) > repeat(Bound , T, T2){
        return repeater_impl<Bound, VIENNACL_TYPELIST2(T,T2) > ();
    }

    template<class Bound, class T, class T2, class T3>
    repeater_impl<Bound, VIENNACL_TYPELIST3(T,T2,T3) > repeat(Bound , T, T2, T3){
        return repeater_impl<Bound, VIENNACL_TYPELIST3(T,T2,T3) > ();
    }


    template<class Bound, class T, class T2, class T3, class T4>
    repeater_impl<Bound, VIENNACL_TYPELIST4(T,T2, T3, T4) > repeat(Bound , T, T2, T3, T4){
        return repeater_impl<Bound, VIENNACL_TYPELIST4(T,T2,T3,T4) > ();
    }


    template<class Bound, class T, class T2, class T3, class T4, class T5>
    repeater_impl<Bound, VIENNACL_TYPELIST5(T,T2, T3, T4, T5) > repeat(Bound , T, T2, T3, T4, T5){
        return repeater_impl<Bound, VIENNACL_TYPELIST5(T,T2,T3,T4,T5) > ();
    }


    template<class T>
    struct get_operations_from_expressions;

    template<class Head, class Tail>
    struct get_operations_from_expressions<typelist<Head,Tail> >{
        typedef typelist<Head, typename get_operations_from_expressions<Tail>::Result> Result;
        typedef typelist<Head, typename get_operations_from_expressions<Tail>::Unrolled> Unrolled;
    };

    template<class Bound, class Operations, class Tail>
    struct get_operations_from_expressions<typelist< repeater_impl<Bound,Operations>,Tail> >{
        typedef typename typelist_utils::fuse<
                                            typename get_operations_from_expressions<Operations>::Result
                                          , typename get_operations_from_expressions<Tail>::Result>::Result Result;
        typedef typename typelist_utils::fuse<
                                            typename typelist_utils::append<typename get_operations_from_expressions<Operations>::Unrolled
                                                                            ,Bound>::Result
                                          , typename get_operations_from_expressions<Tail>::Unrolled>::Result Unrolled;
    };

    template<>
    struct get_operations_from_expressions<NullType>{
        typedef NullType Result;
        typedef NullType Unrolled;
    };


  }
}
#endif // TWEAKING_HPP

