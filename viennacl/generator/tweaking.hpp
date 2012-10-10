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
