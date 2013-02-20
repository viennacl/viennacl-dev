/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

//
// *** System
//
#include <iostream>

//
// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/foreach.hpp>

//
// *** ViennaCL
//

//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS 1
//#define VIENNACL_DEBUG_BUILD
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/generator/custom_operation.hpp"
#include "viennacl/generator/elementwise_modifier.hpp"
#include "viennacl/generator/convenience_typedef.hpp"

using namespace boost::numeric;
using namespace viennacl::generator;


template <class TYPE>
bool readVectorFromFile ( const std::string & filename, boost::numeric::ublas::vector<TYPE> & vec ) {
    std::ifstream file ( filename.c_str() );

    if ( !file ) return false;

    unsigned int size;
    file >> size;

    if ( size > 20000 )  //keep execution times short
        size = 20000;
    vec.resize ( size );

    for ( unsigned int i = 0; i < size; ++i ) {
        TYPE element;
        file >> element;
        vec[i] = element;
    }

    return true;
}

template <typename ScalarType, unsigned int Alignment>
ScalarType diff ( ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType,Alignment> & v2 ) {
    ublas::vector<ScalarType> v2_cpu ( v2.size() );
    viennacl::copy( v2.begin(), v2.end(), v2_cpu.begin() );
    for ( unsigned int i=0; i<v1.size(); ++i ) {
        if ( std::max ( fabs ( v2_cpu[i] ), fabs ( v1[i] ) ) > 0 )
            v2_cpu[i] = fabs ( v2_cpu[i] - v1[i] ) / std::max ( fabs ( v2_cpu[i] ), fabs ( v1[i] ) );
        else
            v2_cpu[i] = 0.0;
    }
    return norm_inf ( v2_cpu );
}

template< typename NumericT, unsigned int Alignment, typename Epsilon >
int test ( Epsilon const& epsilon, std::string vecfile) {
    int retval = EXIT_SUCCESS;

	
    ublas::vector<NumericT> vec;
    ublas::vector<NumericT> vec2;
    ublas::vector<NumericT> vec3;

    NumericT                    cpu_scal = static_cast<NumericT> ( 42.1415 );
    viennacl::scalar<NumericT>  gpu_scal = static_cast<NumericT> ( 42.1415 );

    viennacl::generator::symbolic_vector<0,NumericT,Alignment> symv;
    viennacl::generator::symbolic_vector<1,NumericT,Alignment> symv2;

    viennacl::generator::cpu_symbolic_scalar<1,NumericT> cpu_sym_scal2;
    viennacl::generator::gpu_symbolic_scalar<1,NumericT> gpu_sym_scal2;

    viennacl::generator::cpu_symbolic_scalar<2,NumericT> cpu_sym_scal3;
    viennacl::generator::gpu_symbolic_scalar<2,NumericT> gpu_sym_scal3;

    viennacl::generator::symbolic_vector<3, NumericT, Alignment> symv4;
    
    if ( !readVectorFromFile<NumericT> ( vecfile, vec ) ) {
        std::cout << "Error reading vec file" << std::endl;
        retval = EXIT_FAILURE;
    }


    std::cout << "Running tests for vector of size " << vec.size() << std::endl;
    std::cout << "----- Alignment " << Alignment << " -----" << std::endl;

    viennacl::vector<NumericT,Alignment> vcl_vec ( vec.size() );
    viennacl::vector<NumericT,Alignment> vcl_vec2( vec.size() );
    viennacl::vector<NumericT,Alignment> vcl_vec3( vec.size() );

    vec2 = vec;
    vec3 = 5.0 * vec;
    viennacl::copy ( vec.begin(), vec.end(), vcl_vec.begin() );
    viennacl::copy ( vec2.begin(), vec2.end(), vcl_vec2.begin() );
    viennacl::copy ( vec3.begin(), vec3.end(), vcl_vec3.begin() );

    unsigned int SIZE = vec.size();
    // --------------------------------------------------------------------------

    std::cout << "testing elementwise operations : vec = 1/(1+exp(-vec.*vec2))..." << std::endl;
    for(unsigned int i=0; i < SIZE; ++i){
        vec[i] = 1/(1+exp(-vec[i]*vec2[i]));
    }
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = _1_/(_1_ + math::exp(-element_prod(symv,symv2))), "vec_elementwise_test") ( vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing addition..." << std::endl;
    vec     = ( vec - vec2 );
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv - symv2, "vec_add") ( vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "Testing inplace addition..." << std::endl;
    vec     += vec2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv += symv2, "vec_inplace_add" ) ( vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing substraction..." << std::endl;
    vec     = vec - vec2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv - symv2, "vec_sub" ) ( vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: substraction" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "Testing inplace substraction..." << std::endl;
    vec     -= vec2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv -= symv2, "vec_inplace_sub" ) ( vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace substraction" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    // --------------------------------------------------------------------------

    std::cout << "testing cpu scalar multiplication ..." << std::endl;
    vec     = vec*cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv*cpu_sym_scal2, "vec_cpu_scal_mul") ( vcl_vec, cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: cpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace cpu scalar multiplication ..." << std::endl;
    vec     *= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv *= cpu_sym_scal2, "vec_inplace_cpu_scal_mul" ) ( vcl_vec, cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace cpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing cpu scalar division ..." << std::endl;
    vec     = vec/cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv/cpu_sym_scal2, "vec_cpu_scal_div") ( vcl_vec, cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: cpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace cpu scalar division ..." << std::endl;
    vec     /= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv /= cpu_sym_scal2, "vec_inplace_cpu_scal_div" ) ( vcl_vec, cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace cpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing gpu scalar multiplication ..." << std::endl;
    vec     = vec*cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv*gpu_sym_scal2, "vec_gpu_scal_mul" ) ( vcl_vec, gpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: gpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing cpu and gpu scalar multiplication ..." << std::endl;
    vec     = cpu_scal*vec*cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = cpu_sym_scal2*symv*gpu_sym_scal3, "vec_cpu_gpu_scal_mul" ) ( vcl_vec, cpu_scal, gpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: cpu and gpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    // --------------------------------------------------------------------------

    std::cout << "testing addition scalar multiplication..." << std::endl;
    vec     = vec + cpu_scal*vec2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv + cpu_sym_scal3*symv2, "vec_multiply_add" ) ( vcl_vec, vcl_vec2, cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: addition scalar multiplication" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }


    // --------------------------------------------------------------------------
    std::cout << "testing packed operations..." << std::endl;
    vec     = vec - cpu_scal*(vec2 + vec);
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv = symv - cpu_sym_scal3*(symv2+symv), "vec_packed_operations" ) ( vcl_vec, vcl_vec2,cpu_scal ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: multiple addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    
    // --------------------------------------------------------------------------
    std::cout << "testing multi-line operations..." << std::endl;
    vec3    = vec - cpu_scal*vec2;
    vec     = vec + cpu_scal*vec2;
    viennacl::generator::custom_operation multi_op( symv4 = symv - cpu_sym_scal3 * symv2,
                                                    symv  = symv + cpu_sym_scal3 * symv2,
                                                    "vec_multi_operations" ) ;
    //std::cout << "source: " << multi_op.kernels_source_code() << std::endl;
    viennacl::ocl::enqueue ( multi_op( vcl_vec, vcl_vec2, cpu_scal, vcl_vec3 ) );
    if ( fabs ( diff ( vec, vcl_vec ) ) > epsilon ) {
        std::cout << "# Error at operation: multiple addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( vec, vcl_vec ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    
    return retval;
}


int main() {
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Test :: Vector" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    int retval = EXIT_SUCCESS;

    std::string vecfile ( "../examples/testdata/rhs65025.txt" );

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
    {
        typedef double NumericT;
        NumericT epsilon = 1.0E-4;
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  eps:     " << epsilon << std::endl;
        std::cout << "  numeric: float" << std::endl;
        retval = test<NumericT,1> ( epsilon, vecfile );
        retval &= test<NumericT,4> ( epsilon, vecfile );
        retval &= test<NumericT,16> ( epsilon, vecfile );

        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }
}
