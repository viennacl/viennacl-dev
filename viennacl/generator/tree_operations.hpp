#ifndef VIENNACL_GENERATOR_TREE_OPERATIONS_HPP
#define VIENNACL_GENERATOR_TREE_OPERATIONS_HPP

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

/** @file viennacl/generator/tree_operations.hpp
 *  @brief Functors for modifying the expression tree. Experimental.
 * 
 *  Generator code contributed by Philippe Tillet
 */


#include "viennacl/generator/elementwise_modifier.hpp"
#include "viennacl/generator/meta_tools/typelist.hpp"
namespace viennacl 
{
  namespace generator
  {
    namespace tree_utils 
    {

      /*
      * Count if
      */

      /** @brief Functor for counting the number of elements satisfying a Pred
          @tparam T Tree to scan
          @tparam Pred Pred to be satisfied
      */
      template <class T, template<class> class Pred>
      struct count_if 
      {
        enum { value = Pred<T>::value };
      };


      template<class Head, class Tail, template<class> class Pred>
      struct count_if<typelist<Head,Tail>,Pred>
      {
          enum { value = count_if<Head,Pred>::value + count_if<Tail,Pred>::value };
      };

      template <class T, template<class> class Pred>
      struct count_if<elementwise_modifier<T>, Pred>
      {
        enum { value = Pred<typename T::PRIOR_TYPE>::value + count_if<typename T::PRIOR_TYPE, Pred>::value };
      };

      template<class T, template<class> class Pred>
      struct count_if<inner_prod_impl_t<T>, Pred> 
      {
        enum { value = Pred<inner_prod_impl_t<T> >::value + count_if<T, Pred>::value };
      };


      template<class LHS, class RHS, class OP, template<class> class Pred>
      struct count_if<compound_node<LHS,OP,RHS>,Pred>
      {
        private:
          typedef compound_node<LHS,OP,RHS> T;
          
        public:
          enum { value = Pred<T>::value
                        +  count_if<LHS, Pred>::value
                        +  count_if<RHS, Pred>::value
                };
      };


      /*
      * Count if type
      */


      /** @brief Functor for counting the number of elements equals to the type specified
          @tparam T Tree to scan
          @tparam Searched Type searched for
      */
      template<class T, class Searched>
      struct count_if_type 
      {
        enum { value = 0 };
      };

      template<class Head, class Tail, class Searched>
      struct count_if_type<typelist<Head,Tail>,Searched>
      {
          enum { value = count_if_type<Head,Searched>::value + count_if_type<Tail,Searched>::value };
      };

      template<class T>
      struct count_if_type<T,T> 
      {
        enum { value = 1 };
      };

      template<class T, class Searched>
      struct count_if_type<elementwise_modifier<T>, Searched>
      {
        enum { value = count_if_type<typename T::PRIOR_TYPE, Searched>::value };
      };

      template <class T>
      struct count_if_type<elementwise_modifier<T>, elementwise_modifier<T> >
      {
        enum { value = 1 + count_if_type<typename T::PRIOR_TYPE, elementwise_modifier<T> >::value };
      };

      template <class LHS, class OP, class RHS>
      struct count_if_type<compound_node<LHS, OP, RHS>,
                           compound_node<LHS, OP, RHS> >
      {
        private:
          typedef compound_node<LHS, OP, RHS> T;
        public:
          enum { value = 1 +  count_if_type<LHS, T>::value
                           +  count_if_type<RHS, T>::value
               };
      };

      template <class LHS, class OP, class RHS, class Searched>
      struct count_if_type< compound_node<LHS,OP,RHS>, Searched>
      {
        enum { value = count_if_type<LHS, Searched>::value
                      +  count_if_type<RHS, Searched>::value
             };
      };


      /*
      * Expand
      */


      /** @brief Expand a node on the right */
      template <class LHS, class OP, class RHS_LHS, class RHS_OP, class RHS_RHS>
      struct expand_right 
      {
          typedef compound_node< compound_node<LHS, OP, RHS_LHS>,
                                 RHS_OP,
                                 compound_node<LHS, OP, RHS_RHS> >   Result;
      };

      /** @brief Expand a node on the left */
      template <class LHS_LHS, class LHS_OP, class LHS_RHS, class OP, class RHS>
      struct expand_left 
      {
          typedef compound_node< compound_node<LHS_LHS, OP, RHS>,
                                 LHS_OP,
                                 compound_node<LHS_RHS, OP, RHS> >        Result;
      };

      /** @brief Expands the particular tree
          @tparam T Input tree
      */
      template <class T>
      struct expand 
      {
        typedef T Result;
      };

      template <class T>
      struct expand< elementwise_modifier<T> >
      {
        private:
          typedef typename expand<typename T::PRIOR_TYPE>::Result                 SUB_Result;
        public:
          typedef elementwise_modifier<SUB_Result>    Result;
      };

      template<class T>
      struct expand<inner_prod_impl_t<T> > 
      {
        private:
          typedef typename expand<T>::Result      SUB_Result;
        public:
          typedef inner_prod_impl_t<SUB_Result>   Result;
      };


      template<class LHS,class OP,class RHS>
      struct expand< compound_node<LHS,OP,RHS> >
      {
        typedef compound_node<typename expand<LHS>::Result, OP, typename expand<RHS>::Result>   Result;
      };

      #define make_right_expandable(__OPERATOR1__ , __OPERATOR2__) \
                      template<class LHS, class RHS_LHS, class RHS_RHS>\
                      struct expand< compound_node<LHS, __OPERATOR1__, compound_node<RHS_LHS, __OPERATOR2__, RHS_RHS> > >\
                      {\
                        typedef typename expand_right<typename expand<LHS>::Result\
                                                    , __OPERATOR1__\
                                                    , typename expand<RHS_LHS>::Result\
                                                    , __OPERATOR2__\
                                                    , typename expand<RHS_RHS>::Result\
                                                    >::Result Result;\
                      }

      #define make_left_expandable(__OPERATOR1__ , __OPERATOR2__) \
                      template<class LHS_LHS, class LHS_RHS, class RHS>\
                      struct expand< compound_node< compound_node<LHS_LHS, __OPERATOR2__ , LHS_RHS>\
                                                    , __OPERATOR1__\
                                                    , RHS\
                                                    > >\
                      {\
                        typedef typename expand_left< typename expand<LHS_LHS>::Result\
                                                    , __OPERATOR2__\
                                                    , typename expand<LHS_RHS>::Result\
                                                    , __OPERATOR1__\
                                                    , typename expand<RHS>::Result\
                                                      >	::Result Result;\
                      }

      make_right_expandable ( scal_mul_type,add_type );
      make_right_expandable ( scal_mul_type,sub_type );
      make_left_expandable ( scal_mul_type,add_type );
      make_left_expandable ( scal_mul_type,sub_type );


      #undef make_left_expandable
      #undef make_right_expandable


      ////////////////////////////////
      //////// EXTRACTIF ////////
      ///////////////////////////////


      /** @brief Extracts the types in the tree satisfying a certain predicate.

        @tparam T Tree to scan.
        @tparam Pred Predicate to be satisfied
        @tparam Comparison relation in which the output will be sorted
      */
      template <class T, 
                template<class> class Pred,
                template<class, class> class Comp = typelist_utils::true_comp,
                class TList = NullType>
      struct extract_if 
      {
        private:
          typedef typelist<T,TList>    TypeTrue;
          typedef NullType             TypeFalse;
        public:
          typedef typename get_type_if<TypeTrue, TypeFalse, Pred<T>::value>::Result      Result;
      };

      template <class Head,  class Tail,
                template<class> class Pred,
                template<class, class> class Comp,
                class TList>
      struct extract_if <typelist<Head,Tail>, Pred, Comp, TList>
      {
      private:
	typedef typename extract_if<Head,Pred,Comp,TList>::Result HeadResult;
	typedef typename extract_if<Tail,Pred, Comp, TList>::Result TailResult;
	typedef typename typelist_utils::fuse<TList, HeadResult>::Result       TmpResult1;
      public:
        typedef typename typelist_utils::fuse<TmpResult1, TailResult>::Result  Result;
      };

      
      template <class T,
                template<class> class Pred,
                template<class,class> class Comp,
                class TList>
      struct extract_if<elementwise_modifier<T>, Pred, Comp, TList>
      {
        private:
          typedef typename extract_if<typename T::PRIOR_TYPE, Pred, Comp, TList>::Result         SUB_Result;
        public:
          typedef typename typelist_utils::fuse<TList,SUB_Result>::Result   Result;
      };

      template <class T,
                template<class> class Pred,
                template<class,class> class Comp,
                class TList>
      struct extract_if<inner_prod_impl_t<T>, Pred, Comp, TList > 
      {
        private:
          typedef typename T::LHS LHS;
          typedef typename T::RHS RHS;
          typedef typename extract_if<LHS, Pred, Comp, TList>::Result            LHS_Result;
          typedef typename extract_if<RHS,  Pred, Comp, TList>::Result           RHS_Result;
          typedef typename typelist_utils::fuse<TList, LHS_Result>::Result       TmpResult1;
          typedef typename typelist_utils::fuse<TmpResult1, RHS_Result>::Result  TmpResult2;
          
          typedef TmpResult2                                                                        TypeFalse;
          typedef typename typelist_utils::append<TmpResult2, inner_prod_impl_t<T> >::Result        TypeTrue;
          
        public:
          typedef typename get_type_if<TypeTrue, TypeFalse, Pred< inner_prod_impl_t<T> >::value>::Result   Result;
      };

      template <class LHS, class OP, class RHS,
                template<class> class Pred,
                template<class,class> class Comp,
                class TList>
      struct extract_if< compound_node<LHS, OP, RHS>, Pred, Comp, TList>
      {
        private:
          typedef compound_node<LHS,OP,RHS> T;
          typedef typename extract_if<LHS,Pred,Comp,TList>::Result LHS_Result;
          typedef typename extract_if<RHS,Pred,Comp,TList>::Result RHS_Result;

          typedef typename typelist_utils::fuse< typename typelist_utils::fuse<TList, LHS_Result, Comp>::Result,
                                                 RHS_Result, 
                                                 Comp >::Result     TypeFalse;
          typedef typelist<T, TypeFalse>                                TypeTrue;
        public:
          typedef typename get_type_if<TypeTrue, TypeFalse, Pred<T>::value>::Result       Result;
      };


      ///////////////////////////////


      /** @brief Like extract_if but ignores duplicates.

        @tparam T Tree to scan.
        @tparam Pred Predicate to be satisfied
        @tparam Comparison relation in which the output will be sorted
      */
      template <class T,
                template<class> class Pred,
                template<class, class> class Comp = typelist_utils::true_comp>
      struct extract_if_unique
      {
        typedef typename extract_if<T,Pred,Comp>::Result Tmp;
        typedef typename typelist_utils::no_duplicates<Tmp>::Result Result;
      };

      ///////////////////////////////
      //////// FLIP_TREE  ///////////
      ///////////////////////////////

      template <class OP, bool flip>
      struct invert_flip 
      {
        enum { value = flip };
      };

      template <bool flip>
      struct invert_flip<sub_type, flip> 
      {
        enum { value = !flip };
      };

      template <class OP, bool flip>
      struct flip_operator 
      {
          typedef OP Result;
      };

      template <>
      struct flip_operator<sub_type, true> 
      {
          typedef add_type Result;
      };

      template <>
      struct flip_operator<add_type, true> 
      {
          typedef sub_type Result;
      };

      /** @brief Removes parenthesis keeping flipping coherent with the - signs*/
      template <class T, bool flip = false>
      struct flip_tree 
      {
          typedef T Result;
      };

      template <class T,  bool flip>
      struct flip_tree <elementwise_modifier<T>, flip>
      {
        private:
          typedef typename flip_tree<typename T::PRIOR_TYPE, flip>::Result       SUB_Result;
        public:
          typedef elementwise_modifier<SUB_Result>   Result;
      };

      template <class LHS, class OP, class RHS, bool flip>
      struct flip_tree< compound_node<LHS, OP, RHS>, flip>
      {
        private:
          typedef typename flip_tree<LHS,flip>::Result LHS_Result;
          typedef typename flip_tree<RHS, invert_flip<OP, flip>::value >::Result RHS_Result;

        public:
          typedef compound_node<LHS_Result, typename flip_operator<OP, flip>::Result , RHS_Result> Result;
      };

      ////////////////////////////////
      //////// REMOVE_IF ////////////
      ///////////////////////////////

      template <class OP, class RHS>
      struct handle_unary_minus
      {
        typedef RHS Result;
      };

      template <class RHS>
      struct handle_unary_minus<sub_type, RHS> 
      {
        typedef compound_node<NullType,sub_type,RHS> Result;
      };

      template <class T>
      struct compound_to_simple 
      {
        typedef T Result;
      };

      template <class LHS, class OP>
      struct compound_to_simple<compound_node<LHS, OP, NullType> > 
      {
        typedef LHS Result;
      };

      template <class OP>
      struct compound_to_simple<compound_node<NullType, OP, NullType> > 
      {
        typedef NullType Result;
      };
      
      template <class OP, class RHS>
      struct compound_to_simple<compound_node<NullType, OP, RHS> > 
      {
        typedef typename handle_unary_minus<OP,RHS>::Result Result;
      };

      template <class OP, class RHS, class Enable=void>
      struct get_new_operator 
      {
        typedef OP Result;
      };

      template <class RHS_OP, class RHS_RHS>
      struct get_new_operator <sub_type, compound_node<NullType, RHS_OP, RHS_RHS> >
      {
        typedef RHS_OP Result;
      };

      /** @brief Removes the nodes satisfying a predicate from the tree.

        @tparam T tree to scan.
        @tparam Pred predicate to test.
        @tparam inspect_nested inspect what is nested in products
      */
      template <class T, template<class> class Pred, bool inspect_nested=true>
      struct remove_if 
      {
        typedef typename get_type_if<NullType,T,Pred<T>::value>::Result    Result;
        typedef typename get_type_if<NullType,T,Pred<T>::value>::Result    TmpTree;
      };

      template <class T, template<class> class Pred, bool inspect_nested>
      struct remove_if<elementwise_modifier<T>,Pred,inspect_nested >
      {
        typedef elementwise_modifier<typename remove_if<typename T::PRIOR_TYPE,Pred>::Result> Result;
      };

      template <class LHS, class OP, class RHS, template<class> class Pred, bool inspect_nested>
      struct remove_if<compound_node<LHS,OP,RHS>, Pred, inspect_nested>
      {
        private:
          typedef compound_node<LHS,OP,RHS> T;

          typedef typename remove_if<LHS,Pred,inspect_nested>::TmpTree LHS_TmpTree;
          typedef typename remove_if<RHS,Pred,inspect_nested>::TmpTree RHS_TmpTree;
          typedef compound_node<LHS_TmpTree,OP,RHS_TmpTree> TmpTree0;
      typedef typename get_type_if<TmpTree0,T,! (result_of::is_product_leaf<T>::value || result_of::is_inner_product_leaf<T>::value) || inspect_nested>::Result TmpTree1;

          typedef typename compound_to_simple<typename remove_if<LHS,Pred,inspect_nested>::Result>::Result LHS_Result;
          typedef typename compound_to_simple<typename remove_if<RHS,Pred,inspect_nested>::Result>::Result RHS_Result;
          typedef typename compound_to_simple<compound_node<LHS_Result,
                                                            typename get_new_operator<OP,RHS_TmpTree>::Result,
                                                            RHS_Result
                                                            > >::Result    Result0;
          typedef typename get_type_if<Result0,T,!(result_of::is_product_leaf<T>::value || result_of::is_inner_product_leaf<T>::value) || inspect_nested>::Result Result1;

        public:
          typedef typename get_type_if<NullType, TmpTree1,  Pred<T>::value>::Result    TmpTree;
          typedef typename get_type_if<NullType, Result1,   Pred<T>::value>::Result    Result;
      };
      
    }  // namespace tree_utils
  } // namespace generator
} // namespace viennacl
#endif

