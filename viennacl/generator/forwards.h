#ifndef VIENNACL_GENERATOR_FORWARDS_H
#define VIENNACL_GENERATOR_FORWARDS_H

namespace viennacl{

namespace generator{

class custom_operation;
class symbolic_expression_tree_base;
class symbolic_kernel_argument;
class symbolic_datastructure;


template<typename ScalarType> class vector;
template<typename ScalarType> class scalar;
template<class VCL_MATRIX> class matrix;


namespace code_generation{
class optimization_profile;
}

}

}
#endif // FORWARDS_H
