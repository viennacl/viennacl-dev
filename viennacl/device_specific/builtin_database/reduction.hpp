#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_REDUCTION_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_REDUCTION_HPP_

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

inline database_type<reduction_template::parameters_type> init_reduction()
{
  database_type<reduction_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result);
  devices::accelerator::fallback::add_8B(result);

  devices::cpu::fallback::add_4B(result);
  devices::cpu::fallback::add_8B(result);

  devices::gpu::fallback::add_4B(result);
  devices::gpu::fallback::add_8B(result);

  return result;
}

static database_type<reduction_template::parameters_type> reduction = init_reduction();

template<class NumericT>
reduction_template::parameters_type const & reduction_params(ocl::device const & device)
{
  return get_parameters<NumericT>(reduction, device);
}


}
}
}
#endif
