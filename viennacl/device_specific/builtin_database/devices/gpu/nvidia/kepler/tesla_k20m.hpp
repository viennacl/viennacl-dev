#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_KEPLER_K20M_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_KEPLER_K20M_HPP_

/* =========================================================================
   Copyright (c) 2010-2015, Institute for Microelectronics,
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

#include "viennacl/device_specific/templates/matrix_product_template.hpp"

#include "viennacl/device_specific/templates/row_wise_reduction_template.hpp"

#include "viennacl/device_specific/templates/reduction_template.hpp"

#include "viennacl/device_specific/templates/matrix_axpy_template.hpp"

#include "viennacl/device_specific/templates/vector_axpy_template.hpp"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/builtin_database/common.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{
namespace devices{
namespace gpu{
namespace nvidia{
namespace kepler{
namespace tesla_k20m{

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,2,8,32,8,2,4,FETCH_FROM_LOCAL,FETCH_FROM_GLOBAL_STRIDED,4,16));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,16,16,32,2,1,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,16,32));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,2,8,64,16,1,2,FETCH_FROM_LOCAL,FETCH_FROM_GLOBAL_STRIDED,32,4));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,128,32,1,1,1,16,FETCH_FROM_GLOBAL_CONTIGUOUS,FETCH_FROM_LOCAL,16,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,8,32,16,4,8,4,FETCH_FROM_LOCAL,FETCH_FROM_GLOBAL_STRIDED,8,16));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,32,16,32,8,2,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,16,64));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(4,8,2,4,8,2,8,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_GLOBAL_CONTIGUOUS,0,0));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_product_template::parameters_type(1,128,64,1,4,2,16,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_LOCAL,16,8));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", row_wise_reduction_template::parameters_type(1,2,64,1024,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", row_wise_reduction_template::parameters_type(8,16,8,32768,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", row_wise_reduction_template::parameters_type(1,1,128,2048,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", row_wise_reduction_template::parameters_type(1,32,8,2048,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<reduction_template::parameters_type> & db)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", reduction_template::parameters_type(1,256,4096,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<reduction_template::parameters_type> & db)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", reduction_template::parameters_type(1,128,512,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_axpy_template::parameters_type(1,64,8,128,128,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", matrix_axpy_template::parameters_type(1,32,4,128,128,FETCH_FROM_GLOBAL_CONTIGUOUS));
}

inline void add_8B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", vector_axpy_template::parameters_type(1,256,16384,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::unknown, "Tesla K20m", vector_axpy_template::parameters_type(1,256,16384,FETCH_FROM_GLOBAL_STRIDED));
}

}
}
}
}
}
}
}
}
#endif
