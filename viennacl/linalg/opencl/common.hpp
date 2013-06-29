#ifndef VIENNACL_LINALG_OPENCL_COMMON_HPP_
#define VIENNACL_LINALG_OPENCL_COMMON_HPP_

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

/** @file viennacl/linalg/opencl/common.hpp
    @brief Common implementations shared by OpenCL-based operations
*/

#include <cmath>

#include "viennacl/forwards.h"

namespace viennacl
{
  namespace linalg
  {
    namespace opencl
    {

      namespace detail
      {
        inline std::string op_to_string(op_abs)   { return "abs";   }
        inline std::string op_to_string(op_acos)  { return "acos";  }
        inline std::string op_to_string(op_asin)  { return "asin";  }
        inline std::string op_to_string(op_ceil)  { return "ceil";  }
        inline std::string op_to_string(op_cos)   { return "cos";   }
        inline std::string op_to_string(op_cosh)  { return "cosh";  }
        inline std::string op_to_string(op_exp)   { return "exp";   }
        inline std::string op_to_string(op_fabs)  { return "fabs";  }
        inline std::string op_to_string(op_floor) { return "floor"; }
        inline std::string op_to_string(op_log)   { return "log";   }
        inline std::string op_to_string(op_log10) { return "log10"; }
        inline std::string op_to_string(op_sin)   { return "sin";   }
        inline std::string op_to_string(op_sinh)  { return "sinh";  }
        inline std::string op_to_string(op_sqrt)  { return "sqrt";  }
        inline std::string op_to_string(op_tan)   { return "tan";   }
        inline std::string op_to_string(op_tanh)  { return "tanh";  }
      }

    } //namespace opencl
  } //namespace linalg
} //namespace viennacl


#endif
