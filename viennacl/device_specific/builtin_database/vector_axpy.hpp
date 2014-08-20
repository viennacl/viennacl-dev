#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_VECTOR_AXPY_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_VECTOR_AXPY_HPP_

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"

#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/builtin_database/common.hpp"

#include "viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"

namespace viennacl
{
namespace device_specific
{
namespace builtin_database
{

inline database_type<vector_axpy_template::parameters_type> init_vector_axpy()
{
  database_type<vector_axpy_template::parameters_type> result;

  devices::cpu::fallback::add_4B(result);
  devices::cpu::fallback::add_8B(result);

  devices::gpu::fallback::add_4B(result);
  devices::gpu::fallback::add_8B(result);
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result);
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_8B(result);

  return result;
}

static database_type<vector_axpy_template::parameters_type> vector_axpy = init_vector_axpy();

template<class NumericT>
vector_axpy_template::parameters_type const & vector_axpy_params(ocl::device const & device)
{
  return get_parameters<NumericT>(vector_axpy, device);
}

}
}
}
#endif
