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


//
// *** ViennaCL
//

// #define VIENNACL_DEBUG_ALL
// #define VIENNACL_DEBUG_BUILD
// #define VIENNACL_WITH_UBLAS 1
// #define VIENNACL_DEBUG_CUSTOM_OPERATION
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/generator/custom_operation.hpp"

using namespace boost::numeric;

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

template <typename ScalarType>
ScalarType diff ( ScalarType & s1, viennacl::scalar<ScalarType> & s2 ) 
{
    viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
    if ( s1 != s2 )
        return ( s1 - s2 ) / std::max ( fabs ( s1 ), fabs ( s2 ) );
    return 0;
}

template< typename NumericT,unsigned int Alignment, typename Epsilon >
int test ( Epsilon const& epsilon, std::string vecfile ) {
    int retval = EXIT_SUCCESS;

    viennacl::scalar<NumericT>  vcl_res ( 0 );
    ublas::vector<NumericT> vec;
    ublas::vector<NumericT> vec2;

    NumericT res;

    viennacl::generator::gpu_symbolic_scalar<0,NumericT> symres;
    viennacl::generator::symbolic_vector<1,NumericT,Alignment> symv;
    viennacl::generator::symbolic_vector<2,NumericT,Alignment> symv2;
    viennacl::generator::cpu_symbolic_scalar<3,NumericT> symscal;
    viennacl::generator::cpu_symbolic_scalar<2,NumericT> symscal2;


    if ( !readVectorFromFile<NumericT> ( vecfile, vec ) ) {
        std::cout << "Error reading vec file" << std::endl;
        retval = EXIT_FAILURE;
    }
// 
    std::cout << "Running tests for vector of size " << vec.size() << std::endl;
	std::cout << "----- Alignment " << Alignment << " -----" << std::endl;
// 
    viennacl::vector<NumericT,Alignment> vcl_vec ( vec.size() );
    viennacl::vector<NumericT,Alignment> vcl_vec2 ( vec.size() );
// 
    vec2 = vec;
    viennacl::copy ( vec.begin(), vec.end(), vcl_vec.begin() );
    viennacl::copy ( vec2.begin(), vec2.end(), vcl_vec2.begin() );

//     --------------------------------------------------------------------------

    std::cout << "testing inner product..." << std::endl;
	
    res = ublas::inner_prod ( vec, vec2 );
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation(symres = inner_prod ( symv, symv2 ), "inner_prod") ( vcl_res, vcl_vec, vcl_vec2 ) );
    //std::cout << viennacl::generator::custom_operation(symres = inner_prod ( symv, symv2 ), "inner_prod") .kernels_source_code() << std::endl;
    if ( fabs ( diff ( res, vcl_res ) ) > epsilon ) {
        std::cout << "# Error at operation: inner product" << std::endl;
        std::cout << "  Diff " << fabs ( diff ( res, vcl_res ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inner product division..." << std::endl;
    res = ublas::inner_prod ( vec, vec2 ) /ublas::inner_prod ( vec, vec );
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symres = inner_prod ( symv, symv2 ) /inner_prod ( symv,symv ), "inner_prod_division" ) ( vcl_res, vcl_vec, vcl_vec2 ) );
    if ( fabs ( diff ( res, vcl_res ) ) > epsilon ) {
        std::cout << "# Error at operation: inner_prod_division" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( res, vcl_res ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing scalar / inner product..." << std::endl;
    res = 4/ublas::inner_prod ( vec, vec );
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symres = symscal2/inner_prod ( symv,symv ),"scalar_division" ) ( vcl_res, vcl_vec, 4.0f ) );
    //std::cout << viennacl::generator::custom_operation ( symres = symscal2/inner_prod ( symv,symv ), "scalar_division" ).kernels_source_code() << std::endl;
    if ( fabs ( diff ( res, vcl_res ) ) > epsilon ) {
        std::cout << "# Error at operation: scalar over inner product" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( res, vcl_res ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inner_prod - ( scal - inner_prod ) " << std::endl;
    res = ublas::inner_prod ( vec, vec2 ) - ( 5.0f - inner_prod ( vec,vec2 ) );
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symres = inner_prod ( symv, symv2 ) - ( symscal - inner_prod ( symv,symv2 ) ), "inner_prod_minus_scal_minus_inprod" ) ( vcl_res, vcl_vec, vcl_vec2, 5.0f ) );
    if ( fabs ( diff ( res, vcl_res ) ) > epsilon ) {
        std::cout << "# Error at operation: inner_prod minus ( scal minus inner_prod ) " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( res, vcl_res ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    return retval;
}


int main() {
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Test :: Inner Product" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    int retval = EXIT_SUCCESS;

    std::string vecfile ( "../examples/testdata/rhs65025.txt" );

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
    {
        typedef float NumericT;
        NumericT epsilon = 1.0E-4;
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  eps:     " << epsilon << std::endl;
        std::cout << "  numeric: float" << std::endl;
        retval = test<NumericT,1> ( epsilon, vecfile );
//  		retval = test<NumericT,4> ( epsilon, vecfile, resultfile );
//        retval = test<NumericT,16> ( epsilon, vecfile, resultfile );
        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }
}
