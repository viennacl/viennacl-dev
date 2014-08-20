#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_

#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"


#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/builtin_database/devices/accelerator/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/cpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"
#include "viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi/geforce_gt_540m.hpp"

namespace viennacl
{
namespace device_specific
{
namespace builtin_database
{

inline database_type<matrix_product_template::parameters_type> init_matrix_product_N_N()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());


  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_T_N()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());

  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_N_T()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  devices::gpu::nvidia::fermi::geforce_gt_540m::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());

  return result;
}

inline database_type<matrix_product_template::parameters_type> init_matrix_product_T_T()
{
  database_type<matrix_product_template::parameters_type> result;

  devices::accelerator::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::accelerator::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  devices::cpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::cpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  return result;
}

static database_type<matrix_product_template::parameters_type> matrix_product_N_N = init_matrix_product_N_N();
static database_type<matrix_product_template::parameters_type> matrix_product_T_N = init_matrix_product_T_N();
static database_type<matrix_product_template::parameters_type> matrix_product_N_T = init_matrix_product_N_T();
static database_type<matrix_product_template::parameters_type> matrix_product_T_T = init_matrix_product_T_T();

template<class NumericT>
matrix_product_template::parameters_type const & matrix_product_params(ocl::device const & device, char A_trans, char B_trans)
{
  assert(A_trans=='N' || A_trans=='T');
  assert(B_trans=='N' || B_trans=='T');
  database_type<matrix_product_template::parameters_type> * db;
  if (A_trans=='N' && B_trans=='N')
    db = &matrix_product_N_N;
  else if (A_trans=='T' && B_trans=='N')
    db = &matrix_product_T_N;
  else if (A_trans=='N' && B_trans=='T')
    db = &matrix_product_N_T;
  else
    db = &matrix_product_T_T;
  return get_parameters<NumericT>(*db, device);
}


}
}
}
#endif
