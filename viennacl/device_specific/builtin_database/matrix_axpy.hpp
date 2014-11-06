#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_AXPY_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_AXPY_HPP_

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_470.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell/geforce_gtx_750_ti.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/scrapper.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla/geforce_gtx_260.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands/tahiti.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands/devastator.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler/tesla_k20m.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gtx_580.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands/hawaii.hpp"

#include "viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen/cypress.hpp"

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

inline database_type<matrix_axpy_template::parameters_type> init_matrix_axpy()
{
  database_type<matrix_axpy_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result);
  devices::accelerator::fallback::add_8B(result);

  devices::cpu::fallback::add_4B(result);
  devices::cpu::fallback::add_8B(result);

  devices::gpu::fallback::add_4B(result);
  devices::gpu::fallback::add_8B(result);
  devices::gpu::amd::evergreen::cypress::add_4B(result);
  devices::gpu::amd::evergreen::cypress::add_8B(result);
  devices::gpu::amd::volcanic_islands::hawaii::add_4B(result);
  devices::gpu::amd::volcanic_islands::hawaii::add_8B(result);
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_4B(result);
  devices::gpu::nvidia::fermi::geforce_gtx_580::add_8B(result);
  devices::gpu::nvidia::kepler::tesla_k20m::add_4B(result);
  devices::gpu::nvidia::kepler::tesla_k20m::add_8B(result);
  devices::gpu::amd::southern_islands::tahiti::add_4B(result);
  devices::gpu::amd::southern_islands::tahiti::add_8B(result);
  devices::gpu::amd::northern_islands::devastator::add_4B(result);
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_4B(result);
  devices::gpu::nvidia::tesla::geforce_gtx_260::add_8B(result);
  devices::gpu::amd::northern_islands::scrapper::add_4B(result);
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_4B(result);
  devices::gpu::nvidia::maxwell::geforce_gtx_750_ti::add_8B(result);
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_4B(result);
  devices::gpu::nvidia::fermi::geforce_gtx_470::add_8B(result);

  return result;
}

static database_type<matrix_axpy_template::parameters_type> matrix_axpy = init_matrix_axpy();

template<class NumericT>
matrix_axpy_template::parameters_type const & matrix_axpy_params(ocl::device const & device)
{
  return get_parameters<NumericT>(matrix_axpy, device);
}

}
}
}
#endif
