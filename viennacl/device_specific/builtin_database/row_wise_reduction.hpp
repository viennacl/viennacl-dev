#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_ROW_WISE_REDUCTION_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_ROW_WISE_REDUCTION_HPP_

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"


#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"

namespace viennacl
{
namespace device_specific
{
namespace builtin_database
{

inline database_type<row_wise_reduction_template::parameters_type> init_row_wise_reduction_N()
{
  database_type<row_wise_reduction_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'N'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'N'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'N'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'N'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_8B(result, char_to_type<'N'>());

  return result;
}

inline database_type<row_wise_reduction_template::parameters_type> init_row_wise_reduction_T()
{
  database_type<row_wise_reduction_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'T'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'T'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'T'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'T'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>());
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'T'>());

  return result;
}

static database_type<row_wise_reduction_template::parameters_type> row_wise_reduction_N = init_row_wise_reduction_N();
static database_type<row_wise_reduction_template::parameters_type> row_wise_reduction_T = init_row_wise_reduction_T();

template<class NumericT>
device_specific::row_wise_reduction_template::parameters_type const & row_wise_reduction_params(ocl::device const & device, char A_trans)
{
  assert(A_trans=='N' || A_trans=='T');
  database_type<row_wise_reduction_template::parameters_type> * db;
  if (A_trans)
    db = &row_wise_reduction_T;
  else
    db = &row_wise_reduction_N;
  return get_parameters<NumericT>(*db, device);
}


}
}
}

#endif
