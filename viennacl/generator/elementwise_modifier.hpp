#ifndef VIENNACL_GENERATOR_ELEMENTWISE_MODIFIER_HPP
#define VIENNACL_GENERATOR_ELEMENTWISE_MODIFIER_HPP

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

/** @file viennacl/generator/elementwise_modifier.hpp
 *   @brief Contains the stuffs related to the elementwise_modifier. Experimental.
 * 
 *  Generator code contributed by Philippe Tillet
 */

#include <typeinfo>
#include <string>
#include <algorithm>

#include "viennacl/generator/forwards.h"

namespace viennacl 
{
  namespace generator
  {

    /**
    * @brief Implementation of the elementwise_modifier
    * 
    * @tparam T the underlying expression to modify
    * @tparam U the function returning the modifier's expression
    */
    template<class T>
    struct elementwise_modifier_impl
    {
      protected:
        static std::string modify(std::string expr, std::string const & replacer)
        {
          std::string result(expr);
          size_t pos = 0;
          while( (pos = result.find('X')) != std::string::npos )
          {
            result.replace(pos, 1, '(' + replacer + ')' );
          }

          return result;
        }

      public:
        typedef T PRIOR_TYPE;

        enum { id = -2 };

    };

    template< class T>
    struct elementwise_modifier{
        static std::string name() { return T::name(); }
        static std::string modify(std::string const & replacer) {  return T::modify(replacer); }
    };

#define WRAP_OPENCL_FUNCTION( FUNC_NAME ) \
        template<class T> \
        struct FUNC_NAME ## _modifier : public elementwise_modifier_impl<T>{ \
            static std::string name() \
            { \
                return #FUNC_NAME + T::name(); \
            } \
            static std::string modify(std::string const & replacer) \
            { \
                return elementwise_modifier_impl<T>::modify(#FUNC_NAME "(X)",replacer); \
            } \
        }; \
        \
        \
        template<class T>\
        elementwise_modifier<FUNC_NAME ## _modifier<T> > FUNC_NAME(T){\
            return elementwise_modifier<FUNC_NAME ## _modifier<T> >();\
        }\

    namespace math{


    WRAP_OPENCL_FUNCTION(acos)
    WRAP_OPENCL_FUNCTION(acosh)
    WRAP_OPENCL_FUNCTION(acospi)

    WRAP_OPENCL_FUNCTION(asin)
    WRAP_OPENCL_FUNCTION(asinh)
    WRAP_OPENCL_FUNCTION(asinpi)

    WRAP_OPENCL_FUNCTION(atan)
    WRAP_OPENCL_FUNCTION(atan2)
    WRAP_OPENCL_FUNCTION(atanh)
    WRAP_OPENCL_FUNCTION(atanpi)
    WRAP_OPENCL_FUNCTION(atan2pi)

    WRAP_OPENCL_FUNCTION(cbrt)
    WRAP_OPENCL_FUNCTION(ceil)
    WRAP_OPENCL_FUNCTION(copysign)

    WRAP_OPENCL_FUNCTION(cos)
    WRAP_OPENCL_FUNCTION(cosh)
    WRAP_OPENCL_FUNCTION(cospi)

    WRAP_OPENCL_FUNCTION(erfc)
    WRAP_OPENCL_FUNCTION(erf)
    WRAP_OPENCL_FUNCTION(exp)
    WRAP_OPENCL_FUNCTION(exp2)
    WRAP_OPENCL_FUNCTION(exp10)

    WRAP_OPENCL_FUNCTION(expm1)

    WRAP_OPENCL_FUNCTION(fabs)
    WRAP_OPENCL_FUNCTION(fdim)
    WRAP_OPENCL_FUNCTION(floor)
    WRAP_OPENCL_FUNCTION(fmax)
    WRAP_OPENCL_FUNCTION(fmin)
    WRAP_OPENCL_FUNCTION(fmod)

    WRAP_OPENCL_FUNCTION(fract)
    WRAP_OPENCL_FUNCTION(frexp)

    WRAP_OPENCL_FUNCTION(ilogb)
    WRAP_OPENCL_FUNCTION(ldexp)

    WRAP_OPENCL_FUNCTION(lgamma)
    WRAP_OPENCL_FUNCTION(lgamma_r)

    WRAP_OPENCL_FUNCTION(log)
    WRAP_OPENCL_FUNCTION(log2)
    WRAP_OPENCL_FUNCTION(log10)
    WRAP_OPENCL_FUNCTION(log1p)
    WRAP_OPENCL_FUNCTION(logb)

    WRAP_OPENCL_FUNCTION(rint)
    WRAP_OPENCL_FUNCTION(round)
    WRAP_OPENCL_FUNCTION(rsqrt)
    WRAP_OPENCL_FUNCTION(sin)
    WRAP_OPENCL_FUNCTION(sinh)
    WRAP_OPENCL_FUNCTION(sinpi)
    WRAP_OPENCL_FUNCTION(sqrt)
    WRAP_OPENCL_FUNCTION(tan)
    WRAP_OPENCL_FUNCTION(tanh)
    WRAP_OPENCL_FUNCTION(tanpi)
    WRAP_OPENCL_FUNCTION(tgamma)
    WRAP_OPENCL_FUNCTION(trunk)


    WRAP_OPENCL_FUNCTION(native_cos)
    WRAP_OPENCL_FUNCTION(half_cos)

    WRAP_OPENCL_FUNCTION(native_exp)
    WRAP_OPENCL_FUNCTION(half_exp)

    WRAP_OPENCL_FUNCTION(native_exp2)
    WRAP_OPENCL_FUNCTION(half_exp2)

    WRAP_OPENCL_FUNCTION(native_exp10)
    WRAP_OPENCL_FUNCTION(half_exp10)

    WRAP_OPENCL_FUNCTION(native_log)
    WRAP_OPENCL_FUNCTION(half_log)

    WRAP_OPENCL_FUNCTION(native_log2)
    WRAP_OPENCL_FUNCTION(half_log2)

    WRAP_OPENCL_FUNCTION(native_log10)
    WRAP_OPENCL_FUNCTION(half_log10)

    WRAP_OPENCL_FUNCTION(native_recip)
    WRAP_OPENCL_FUNCTION(half_recip)

    WRAP_OPENCL_FUNCTION(native_rsqrt)
    WRAP_OPENCL_FUNCTION(half_rsqrt)

    WRAP_OPENCL_FUNCTION(native_sin)
    WRAP_OPENCL_FUNCTION(half_sin)

    WRAP_OPENCL_FUNCTION(native_sqrt)
    WRAP_OPENCL_FUNCTION(half_sqrt)

    WRAP_OPENCL_FUNCTION(native_tan)
    WRAP_OPENCL_FUNCTION(half_tan)

    }
#undef WRAP_OPENCL_FUNCTION
  }
}

#endif

