#ifndef VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_
#define VIENNACL_DEVICE_SPECIFIC_BUILTIN_DATABASE_MATRIX_PRODUCT_HPP_

#include "viennacl/ocl/device_utils.hpp"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/builtin_database/devices/gpu/fallback.hpp"

namespace viennacl{
namespace device_specific{
namespace builtin_database{


inline database_type<matrix_product_template::parameters> init_matrix_product_NN()
{
  database_type<matrix_product_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'N'>());

  return result;
}

inline database_type<matrix_product_template::parameters> init_matrix_product_TN()
{
  database_type<matrix_product_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'N'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'N'>());

  return result;
}

inline database_type<matrix_product_template::parameters> init_matrix_product_NT()
{
  database_type<matrix_product_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'N'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'N'>(), char_to_type<'T'>());

  return result;
}

inline database_type<matrix_product_template::parameters> init_matrix_product_TT()
{
  database_type<matrix_product_template::parameters> result;

  devices::gpu::fallback::add_4B(result, char_to_type<'T'>(), char_to_type<'T'>());
  devices::gpu::fallback::add_8B(result, char_to_type<'T'>(), char_to_type<'T'>());

  return result;
}

static database_type<matrix_product_template::parameters> matrix_product_NN = init_matrix_product_NN();
static database_type<matrix_product_template::parameters> matrix_product_TN = init_matrix_product_TN();
static database_type<matrix_product_template::parameters> matrix_product_NT = init_matrix_product_NT();
static database_type<matrix_product_template::parameters> matrix_product_TT = init_matrix_product_TT();

template<class T>
matrix_product_template::parameters const & matrix_product_params(ocl::device const & device, char A_trans, char B_trans)
{
  assert(A_trans=='N' || A_trans=='T');
  assert(B_trans=='N' || B_trans=='T');
  database_type<matrix_product_template::parameters> * db;
  if(A_trans=='N' && B_trans=='N')
    db = &matrix_product_NN;
  else if(A_trans=='T' && B_trans=='N')
    db = &matrix_product_TN;
  else if(A_trans=='N' && B_trans=='T')
    db = &matrix_product_NT;
  else
    db = &matrix_product_TT;
  return get_parameters<T>(*db, device);
}


}
}
}
#endif
