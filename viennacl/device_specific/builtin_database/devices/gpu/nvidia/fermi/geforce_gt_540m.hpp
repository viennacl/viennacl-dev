#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_FERMI_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_DEVICES_GPU_NVIDIA_FERMI_HPP_

#include "viennacl/device_specific/templates/matrix_axpy_template.hpp"

#include "viennacl/device_specific/templates/matrix_product_template.hpp"

#include "viennacl/device_specific/templates/row_wise_reduction_template.hpp"

#include "viennacl/device_specific/templates/vector_axpy_template.hpp"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/builtin_database/common.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{
namespace devices{
namespace gpu{
namespace nvidia{
namespace fermi{
namespace geforce_gt_540m{

inline void add_4B(database_type<matrix_product_template::parameters> & db, char_to_type<'T'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", matrix_product_template::parameters(1, 16, 16, 8, 4, 1, 8, 1, 1, 16, 8));
}

inline void add_4B(database_type<matrix_product_template::parameters> & db, char_to_type<'N'>, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", matrix_product_template::parameters(1, 16, 16, 16, 8, 1, 4, 1, 1, 32, 8));
}

inline void add_4B(database_type<matrix_product_template::parameters> & db, char_to_type<'N'>, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", matrix_product_template::parameters(1, 8, 16, 16, 8, 1, 4, 1, 1, 16, 8));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters> & db, char_to_type<'T'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", row_wise_reduction_template::parameters(1, 1, 256, 1024));
}

inline void add_8B(database_type<row_wise_reduction_template::parameters> & db, char_to_type<'N'>)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", row_wise_reduction_template::parameters(1, 1024, 1, 16));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters> & db, char_to_type<'T'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", row_wise_reduction_template::parameters(1, 1, 256, 2048));
}

inline void add_4B(database_type<row_wise_reduction_template::parameters> & db, char_to_type<'N'>)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", row_wise_reduction_template::parameters(1, 32, 16, 128));
}

inline void add_8B(database_type<matrix_axpy_template::parameters> & db)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", matrix_axpy_template::parameters(1, 1, 128, 32, 16, 1));
}

inline void add_4B(database_type<matrix_axpy_template::parameters> & db)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", matrix_axpy_template::parameters(1, 1, 128, 4, 4, 0));
}

inline void add_8B(database_type<vector_axpy_template::parameters> & db)
{
  db.add_8B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", vector_axpy_template::parameters(1, 512, 2048, 0));
}

inline void add_4B(database_type<vector_axpy_template::parameters> & db)
{
  db.add_4B(nvidia_id, CL_DEVICE_TYPE_GPU, ocl::fermi, "GeForce GT 540M", vector_axpy_template::parameters(1, 512, 2048, 1));
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
