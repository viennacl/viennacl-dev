#ifndef VIENNACL_GENERATOR_FORWARDS_H
#define VIENNACL_GENERATOR_FORWARDS_H

namespace viennacl{

namespace generator{

class custom_operation;
class infos_base;
class assign_type;
class add_type;
class inplace_add_type;
class sub_type;
class inplace_sub_type;
class scal_mul_type;
class inplace_scal_mul_type;
class scal_div_type;
class inplace_scal_div_type;
class elementwise_prod_type;
class elementwise_div_type;
class trans_type;
template<class REDUCE_TYPE>
class prod_type;

class unary_sub_type;

template<typename ScalarType>
class dummy_vector;

template<typename ScalarType>
class dummy_scalar;

template<class VCL_MATRIX>
class dummy_matrix;

}

}
#endif // FORWARDS_H
