#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_AMD_SOUTHERN_ISLANDS_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_AMD_SOUTHERN_ISLANDS_HPP_

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
namespace amd{
namespace southern_islands{
namespace tahiti{

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,32,16,8,1,1,16,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,16,16));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,16,16,16,4,1,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,4,64));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(2,8,2,16,4,2,4,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_GLOBAL_CONTIGUOUS,0,0));
}

inline void add_8B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,16,4,4,4,2,4,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_LOCAL,4,16));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'T'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,8,32,32,4,1,4,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,32,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,8,32,32,4,1,2,FETCH_FROM_LOCAL,FETCH_FROM_LOCAL,32,8));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,64,32,4,4,2,4,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_LOCAL,16,16));
}

inline void add_4B(database_type<matrix_product_template::parameters_type> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_product_template::parameters_type(1,128,32,2,2,1,8,FETCH_FROM_GLOBAL_STRIDED,FETCH_FROM_LOCAL,32,8));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", row_wise_reduction_template::parameters_type(1,1,128,2048,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", row_wise_reduction_template::parameters_type(2,16,16,1024,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'T'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", row_wise_reduction_template::parameters_type(1,1,128,2048,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters_type> & db, char_to_type<'N'>)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", row_wise_reduction_template::parameters_type(4,8,16,16384,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<reduction_template::parameters_type> & db)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", reduction_template::parameters_type(1,256,4096,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<reduction_template::parameters_type> & db)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", reduction_template::parameters_type(1,256,8192,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_axpy_template::parameters_type(1,128,2,128,2,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<matrix_axpy_template::parameters_type> & db)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", matrix_axpy_template::parameters_type(1,128,2,64,4,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_8B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_8B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", vector_axpy_template::parameters_type(1,64,256,FETCH_FROM_GLOBAL_STRIDED));
}

inline void add_4B(database_type<vector_axpy_template::parameters_type> & db)
{
  db.add_4B(amd_id, CL_DEVICE_TYPE_GPU, ocl::southern_islands, "Tahiti", vector_axpy_template::parameters_type(1,64,512,FETCH_FROM_GLOBAL_STRIDED));
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
