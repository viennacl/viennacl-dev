#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_VECTOR_AXPY_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_VECTOR_AXPY_HPP_

#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/builtin_database/common.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{

inline database_type<vector_axpy_template::parameters> init_vector_axpy()
{
  database_type<vector_axpy_template::parameters> result;

  devices::gpu::fallback::add_4B(result);

  return result;
}

static database_type<vector_axpy_template::parameters> vector_axpy = init_vector_axpy();

template<class T>
vector_axpy_template::parameters const & vector_axpy_params(ocl::device const & device)
{
  return get_parameters<T>(vector_axpy, device);
}


}
}
}
#endif
