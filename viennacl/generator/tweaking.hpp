#ifndef VIENNACL_GENERATOR_TWEAKING_HPP
#define VIENNACL_GENERATOR_TWEAKING_HPP

#include "viennacl/generator/meta_tools/typelist.hpp"

namespace viennacl
{
  namespace generator
  {

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
    };

    template<class Bound, class Operations, class Tail>
    struct get_operations_from_expressions<typelist< repeater_impl<Bound,Operations>,Tail> >{
        typedef typename typelist_utils::fuse<typename get_operations_from_expressions<Operations>::Result, typename get_operations_from_expressions<Tail>::Result>::Result Result;
    };

    template<>
    struct get_operations_from_expressions<NullType>{
        typedef NullType Result;
    };

    template<class T>
    struct unroll_repeaters;

    template<class Head, class Tail>
    struct unroll_repeaters<typelist<Head,Tail> >{
        typedef typelist<Head, typename unroll_repeaters<Tail>::Result> Result;
    };

    template<class Bound, class Operations, class Tail>
    struct unroll_repeaters<typelist< repeater_impl<Bound,Operations>,Tail> >{
        typedef typename typelist_utils::fuse<typename get_operations_from_expressions<Operations>::Result, typename unroll_repeaters<Tail>::Result>::Result ResultTmp;
        typedef typename typelist_utils::append<ResultTmp,Bound>::Result Result;
    };

    template<>
    struct unroll_repeaters<NullType>{
        typedef NullType Result;
    };


  }
}
#endif // TWEAKING_HPP
