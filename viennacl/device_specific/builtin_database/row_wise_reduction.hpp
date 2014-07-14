#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_ROW_WISE_REDUCTION_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_ROW_WISE_REDUCTION_HPP_

#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{


inline database_type<row_wise_reduction_template::parameters> init_row_wise_reduction_N()
{
  database_type<row_wise_reduction_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>());

  return result;
}

inline database_type<row_wise_reduction_template::parameters> init_row_wise_reduction_T()
{
  database_type<row_wise_reduction_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>());

  return result;
}

static database_type<row_wise_reduction_template::parameters> row_wise_reduction_N = init_row_wise_reduction_N();
static database_type<row_wise_reduction_template::parameters> row_wise_reduction_T = init_row_wise_reduction_T();

template<class T>
device_specific::row_wise_reduction_template::parameters const & row_wise_reduction_params(ocl::device const & device, char A_trans)
{
  assert(A_trans=='N' || A_trans=='T');
  database_type<row_wise_reduction_template::parameters> * db;
  if(A_trans)
    db = &row_wise_reduction_T;
  else
    db = &row_wise_reduction_N;
  return get_parameters<T>(*db, device);
}


}
}
}
#endif
