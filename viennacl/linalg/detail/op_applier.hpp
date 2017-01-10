#ifndef VIENNACL_LINALG_DETAIL_OP_APPLIER_HPP
#define VIENNACL_LINALG_DETAIL_OP_APPLIER_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/linalg/detail/op_applier.hpp
 *
 * @brief Defines the action of certain unary and binary operators and its arguments (for host execution).
*/

#include "viennacl/forwards.h"
#include <cmath>

namespace viennacl
{
namespace linalg
{
namespace detail
{

/** @brief Worker class for decomposing expression templates.
  *
  * @tparam A    Type to which is assigned to
  * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
  @ @tparam T    Right hand side of the assignment
*/
template<typename OpT>
struct op_applier
{
  typedef typename OpT::ERROR_UNKNOWN_OP_TAG_PROVIDED    error_type;
};

/** \cond */
template<>
struct op_applier<op_element_binary<op_prod> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x * y; }
};

template<>
struct op_applier<op_element_binary<op_div> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x / y; }
};

template<>
struct op_applier<op_element_binary<op_pow> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = std::pow(x, y); }
};

#define VIENNACL_MAKE_UNARY_OP_APPLIER(funcname)  \
template<> \
struct op_applier<op_element_unary<op_##funcname> > \
{ \
  template<typename T> \
  static void apply(T & result, T const & x) { using namespace std; result = static_cast<T>(funcname(x)); } \
}

VIENNACL_MAKE_UNARY_OP_APPLIER(abs);
VIENNACL_MAKE_UNARY_OP_APPLIER(acos);
VIENNACL_MAKE_UNARY_OP_APPLIER(acosh);
VIENNACL_MAKE_UNARY_OP_APPLIER(asin);
VIENNACL_MAKE_UNARY_OP_APPLIER(asinh);
VIENNACL_MAKE_UNARY_OP_APPLIER(atan);
VIENNACL_MAKE_UNARY_OP_APPLIER(atanh);
VIENNACL_MAKE_UNARY_OP_APPLIER(ceil);
VIENNACL_MAKE_UNARY_OP_APPLIER(cos);
VIENNACL_MAKE_UNARY_OP_APPLIER(cosh);
VIENNACL_MAKE_UNARY_OP_APPLIER(erf);
VIENNACL_MAKE_UNARY_OP_APPLIER(erfc);
VIENNACL_MAKE_UNARY_OP_APPLIER(exp);
VIENNACL_MAKE_UNARY_OP_APPLIER(exp2);
VIENNACL_MAKE_UNARY_OP_APPLIER(fabs);
VIENNACL_MAKE_UNARY_OP_APPLIER(floor);
VIENNACL_MAKE_UNARY_OP_APPLIER(log);
VIENNACL_MAKE_UNARY_OP_APPLIER(log2);
VIENNACL_MAKE_UNARY_OP_APPLIER(log10);
VIENNACL_MAKE_UNARY_OP_APPLIER(round);
VIENNACL_MAKE_UNARY_OP_APPLIER(sin);
VIENNACL_MAKE_UNARY_OP_APPLIER(sinh);
VIENNACL_MAKE_UNARY_OP_APPLIER(sqrt);
VIENNACL_MAKE_UNARY_OP_APPLIER(tan);
VIENNACL_MAKE_UNARY_OP_APPLIER(tanh);
VIENNACL_MAKE_UNARY_OP_APPLIER(trunc);

#undef VIENNACL_MAKE_UNARY_OP_APPLIER

template<>
struct op_applier<op_element_unary<op_exp10> >
{
  template<typename T>
  static void apply(T & result, T const & x) { using namespace std; result = std::exp(x*T(2.302585092994045684017991454684364207601101488628772976033)); }
};

template<>
struct op_applier<op_element_unary<op_rsqrt> >
{
  template<typename T>
  static void apply(T & result, T const & x) { using namespace std; result = std::pow(x, T(-0.5)); }
};

template<>
struct op_applier<op_element_unary<op_sign> >
{
  template<typename T>
  static void apply(T & result, T const & x) { using namespace std; result = (x > T(0)) ? T(1) : (x < T(0) ? T(-1) : T(0)); }
};

/** \endcond */

}
}
}

#endif // VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
