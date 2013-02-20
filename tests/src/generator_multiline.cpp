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

// #define VIENNACL_DEBUG_CUSTOM_OPERATION
//#define VIENNACL_DEBUG_BUILD

//


// *** Boost
//
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>



//
// *** ViennaCL
//

// #define VIENNACL_DEBUG_ALL
// #define VIENNACL_WITH_UBLAS 1
// #define VIENNACL_DEBUG_CUSTOM_OPERATION
//#define VIENNACL_DEBUG_BUILD

#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"
#include "viennacl/generator/custom_operation.hpp"

//
// -------------------------------------------------------------
//
using namespace boost::numeric;
using namespace viennacl::generator;
//
// -------------------------------------------------------------
//

template <typename ScalarType, unsigned int Alignment>
ScalarType diff ( ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType,Alignment> & v2 ) {
    ublas::vector<ScalarType> v2_cpu ( v2.size() );
    viennacl::copy( v2.begin(), v2.end(), v2_cpu.begin() );
    for ( unsigned int i=0; i<v1.size(); ++i ) {
        if ( std::max ( std::abs ( v2_cpu[i] ), std::abs ( v1[i] ) ) > 0 )
            v2_cpu[i] = std::abs ( v2_cpu[i] - v1[i] ) / std::max ( std::abs ( v2_cpu[i] ), std::abs ( v1[i] ) );
        else
            v2_cpu[i] = 0.0;
    }
    return norm_inf ( v2_cpu );
}

template< typename NumericT,  typename F,typename F2, unsigned int Alignment, typename Epsilon >
int test ( Epsilon const& epsilon ) {
    int retval = EXIT_SUCCESS;
    static const unsigned int SIZE = 5;//027;
    // --------------------------------------------------------------------------
    NumericT scalar = 2;
    ublas::vector<NumericT> v0(SIZE);
    ublas::vector<NumericT> v1(SIZE);
    ublas::vector<NumericT> v2(SIZE);
    ublas::vector<NumericT> v3(SIZE);
    for ( unsigned int i = 0; i < SIZE; ++i ){
        v0( i ) = i;
        v1( i ) = i+1;
        v2( i ) = i+2;
        v3( i ) = i+3;
    }
    ublas::matrix<NumericT,F2> matrix ( v0.size(), v1.size() );
    for ( unsigned int i = 0; i < matrix.size1(); ++i )
        for ( unsigned int j = 0; j < matrix.size2(); ++j )
            matrix ( i,j ) = i+j;


    std::cout << "----- Alignment " << Alignment << " -----" << std::endl;

    viennacl::scalar<NumericT> vcl_scalar(scalar);
    viennacl::vector<NumericT,Alignment> vcl_v0 (v0.size());
    viennacl::vector<NumericT,Alignment> vcl_v1 (v1.size());
    viennacl::vector<NumericT,Alignment> vcl_v2 (v2.size());
    viennacl::vector<NumericT,Alignment> vcl_v3 (v3.size());
    viennacl::matrix<NumericT, F, Alignment> vcl_matrix ( matrix.size1(), matrix.size2() );

    viennacl::copy(v0,vcl_v0);
    viennacl::copy(v1,vcl_v1);
    viennacl::copy(v2,vcl_v2);
    viennacl::copy(v3,vcl_v3);

    viennacl::copy(matrix,vcl_matrix);

    // --------------------------------------------------------------------------

    symbolic_vector<0,NumericT> symv0;
    symbolic_vector<1,NumericT> symv1;
    symbolic_vector<2,NumericT> symv2;
    symbolic_vector<3,NumericT> symv3;

    symbolic_matrix<1,NumericT,F,Alignment> symm1;
    symbolic_matrix<3,NumericT,F,Alignment> symm3;

    gpu_symbolic_scalar<3,NumericT> syms3;
    gpu_symbolic_scalar<4,NumericT> syms4;

    cpu_symbolic_scalar<5,NumericT> sym_bound;
    cpu_symbolic_scalar<6,NumericT> sym_bound2;

    // --------------------------------------------------------------------------


    float bound1 = 2;

    std::cout << "Testing repetition..." << std::endl;
    v0 = v1 + v2;
    for(unsigned int i = 0 ; i< bound1 ; ++i){
        v2 = scalar*v2;
    }
    v3 = v0 + v2;
    viennacl::ocl::enqueue(viennacl::generator::custom_operation (symv0 = symv1 + symv2
                                                                    ,viennacl::generator::repeat(sym_bound,symv2 = syms4 * symv2)
                                                                    ,symv3 = symv0 + symv2
                                                                    ,"Multiline1") ( vcl_v0, vcl_v1, vcl_v2, vcl_v3, vcl_scalar, bound1));
    viennacl::ocl::get_queue().finish();

    if ( diff(v0, vcl_v0) > epsilon
         || diff (v2,vcl_v2) > epsilon
         || diff (v3,vcl_v3) > epsilon){
        std::cout << "# Error at operation:repetition" << std::endl;
        std::cout << "  diff0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff2: " << std::abs ( diff ( v2, vcl_v2 ) ) << std::endl;
        std::cout << "  diff3: " << std::abs ( diff ( v2, vcl_v2 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "Testing nested repetition..." << std::endl;

    float bound2 = 1;
    v0 = v1 + v2;
    for(unsigned int i = 0 ; i< bound1 ; ++i){
        v2 = scalar*v2;
        for(unsigned int j = 0 ; j < bound2 ; ++j){
            v3 = v3 - v2;
        }
    }
    v1 = v2 + v3;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv0 = symv1 + symv2
                                                                    ,viennacl::generator::repeat(sym_bound,symv2 = syms4 * symv2
                                                                                                          , viennacl::generator::repeat(sym_bound2,symv3 = symv3 - symv2))
                                                                    ,symv1 = symv2 + symv3
                                                                    ,"Multiline2") ( vcl_v0, vcl_v1, vcl_v2, vcl_v3, vcl_scalar, bound1,bound2) );
    viennacl::ocl::get_queue().finish();

    if ( diff(v0, vcl_v0) > epsilon
         || diff(v1, vcl_v1) > epsilon
         || diff (v2,vcl_v2) > epsilon
         || diff (v3,vcl_v3) > epsilon ) {
        std::cout << "# Error at operation: Nested repetition" << std::endl;
        std::cout << "  diff0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff1: " << std::abs ( diff ( v1, vcl_v1 ) ) << std::endl;
        std::cout << "  diff2: " << std::abs ( diff ( v2, vcl_v2 ) ) << std::endl;
        std::cout << "  diff3: " << std::abs ( diff ( v3, vcl_v3 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();

    std::cout << "Testing V0=V1+V2 ; s=inner_prod(V0,V1) ..." << std::endl;

    v0 = v1 + v2;
    scalar = ublas::inner_prod(v0,v1);

    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv0 = symv1 + symv2
                                                                    ,syms3= inner_prod(symv0,symv1)
                                                                    ,"Multiline3") ( vcl_v0, vcl_v1, vcl_v2, vcl_scalar) );


    viennacl::ocl::get_queue().finish();

    if ( diff (v0,vcl_v0 ) > epsilon
         || std::abs( scalar - static_cast<NumericT>(vcl_scalar) ) / std::max<NumericT>(scalar, vcl_scalar) > epsilon){
        std::cout << "# Error at operation: V0=V1+V2 ; s=inner_prod(V0,V1) " << std::endl;
        std::cout << "  diff_vec: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_scal: " << std::abs ( scalar - vcl_scalar ) << std::endl;
        std::cout << "  scalar: " << scalar << std::endl;
        std::cout << "  vcl_scalar: " << vcl_scalar << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();


    std::cout << "Testing  s=inner_prod(V0,V1) ; V0=V1+V2 ..." << std::endl;
    scalar = ublas::inner_prod(v0,v1);
    v0 = v1 + v2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (syms3= inner_prod(symv0,symv1)
                                                                    ,symv0 = symv1 + symv2
                                                                    ,"Multiline4") ( vcl_v0, vcl_v1, vcl_v2, vcl_scalar) );


    viennacl::ocl::get_queue().finish();
    if ( diff(v0, vcl_v0) > epsilon
         || std::abs( scalar - static_cast<NumericT>(vcl_scalar) ) / std::max<NumericT>(scalar, vcl_scalar) > epsilon){
        std::cout << "# Error at operation: s=inner_prod(V0,V1) ; V0=V1+V2  ... " << std::endl;
        std::cout << "  diff_vec: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_scal: " << std::abs ( scalar - vcl_scalar ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();


    std::cout << "Testing V0=V1+V2 ; V1=prod(M,V0) ..." << std::endl;

    v0 = v1 + v2;
    v1 = ublas::prod(matrix,v0);

    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv0 = symv1 + symv2
                                                                     ,symv1= prod(symm3,symv0)
                                                                    ,"Multiline5") ( vcl_v0, vcl_v1, vcl_v2, vcl_matrix) );


    viennacl::ocl::get_queue().finish();

    if ( diff(v0, vcl_v0) > epsilon
         || diff(v1, vcl_v1) > epsilon){
        std::cout << "# Error at operation: V0=V1+V2 ; V1=prod(M,V0) " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_vec1: " << std::abs ( diff ( v1, vcl_v1 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();


    std::cout << "Testing  V1=prod(M,V0) ; V0=V1+V2 ..." << std::endl;
    v1 = ublas::prod(matrix,v0);
    v0 = v1 + v2;

    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv1= prod(symm3,symv0)
                                                                    ,symv0 = symv1 + symv2
                                                                    ,"Multiline6") ( vcl_v0, vcl_v1, vcl_v2, vcl_matrix) );


    viennacl::ocl::get_queue().finish();

    if ( diff(v0, vcl_v0) > epsilon
         || diff(v1, vcl_v1) > epsilon){
        std::cout << "# Error at operation:  V1=prod(M,V0) ; V0=V1+V2 " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_vec1: " << std::abs ( diff ( v1, vcl_v1 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();


    std::cout << "Testing V0 = M*V2 * ip(V2,V3) ; V2 = V0*s ..." << std::endl;
    v0 = ublas::prod(matrix, v2)*ublas::inner_prod(v2,v3);
    v2 = v0*scalar;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv0 = prod(symm1,symv2)*inner_prod(symv2,symv3)
                                                                    ,symv2 = symv0*syms4
                                                                    ,"Multiline7") ( vcl_v0, vcl_matrix, vcl_v2, vcl_v3, vcl_scalar) );


    viennacl::ocl::get_queue().finish();
    if ( diff(v0, vcl_v0) > epsilon
         || diff(v2, vcl_v2) > epsilon){
        std::cout << "# Error at operation: V0 = M*V2 * ip(V2,V3) ; V2 = V0*s ... " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_vec2: " << std::abs ( diff ( v2, vcl_v2 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();



    std::cout << "Testing V2 = V0*s ; V0 = M*V2 * ip(V2,V3) ..." << std::endl;
    v2 = v0*scalar;
    v0 = ublas::prod(matrix, v2)*ublas::inner_prod(v2,v3);
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv2 = symv0*syms4
                                                                    ,symv0 = prod(symm1,symv2)*inner_prod(symv2,symv3)
                                                                    ,"Multiline8") ( vcl_v0, vcl_matrix, vcl_v2, vcl_v3, vcl_scalar) );


    viennacl::ocl::get_queue().finish();
    if ( diff(v0, vcl_v0) > epsilon
         || diff(v2, vcl_v2) > epsilon){
        std::cout << "# Error at operation: V2 = V0*s ; V0 = M*V2 * ip(V2,V3) ... " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( v0, vcl_v0 ) ) << std::endl;
        std::cout << "  diff_vec2: " << std::abs ( diff ( v2, vcl_v2 ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();

    std::cout << "Testing dX = sigma * (X - Y) ; " 
              << "        dY = R * X - Y - X * Z;"
              << "        dZ = X * Y - b * Z;" << std::endl;
              
    for ( unsigned int i = 0; i < SIZE; ++i ){
        v0( i ) = i;
        v1( i ) = i+1;
        v2( i ) = i+2;
        v3( i ) = i+3;
    }

    viennacl::copy(v0,vcl_v0);
    viennacl::copy(v1,vcl_v1);
    viennacl::copy(v2,vcl_v2);
    viennacl::copy(v3,vcl_v3);
              
    ublas::vector<NumericT> dv0(v0);
    ublas::vector<NumericT> dv1(v1);
    ublas::vector<NumericT> dv2(v2);
    NumericT sigma = 3.1415;
    NumericT b = 2.7182;
    dv0 = sigma * (v1 - v0);
    dv1 = boost::numeric::ublas::element_prod(v3, v0) - v1 - boost::numeric::ublas::element_prod(v0, v2);
    dv2 = boost::numeric::ublas::element_prod(v0, v1) - b * v2;
    

    symbolic_vector<0,NumericT> sym_dX;
    symbolic_vector<1,NumericT> sym_dY;
    symbolic_vector<2,NumericT> sym_dZ;
    
    symbolic_vector<3,NumericT> sym_X;
    symbolic_vector<4,NumericT> sym_Y;
    symbolic_vector<5,NumericT> sym_Z;
    
    symbolic_vector<6,NumericT> sym_R;

    cpu_symbolic_scalar<7,NumericT> sym_sigma;
    cpu_symbolic_scalar<8,NumericT> sym_b;
    
    viennacl::vector<NumericT,Alignment> vcl_dX (vcl_v0);
    viennacl::vector<NumericT,Alignment> vcl_dY (vcl_v1);
    viennacl::vector<NumericT,Alignment> vcl_dZ (vcl_v2);
    viennacl::vector<NumericT,Alignment> vcl_X (vcl_v0);
    viennacl::vector<NumericT,Alignment> vcl_Y (vcl_v1);
    viennacl::vector<NumericT,Alignment> vcl_Z (vcl_v2);
    
    viennacl::vector<NumericT,Alignment> vcl_R (vcl_v3);
    
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( sym_dX = sym_sigma * (sym_Y - sym_X)
                                                                    ,sym_dY = element_prod(sym_R, sym_X) - sym_Y - element_prod(sym_X, sym_Z)
                                                                    ,sym_dZ = element_prod(sym_X, sym_Y) - sym_b * sym_Z
                                                                    ,"Lorenz") ( vcl_dX, vcl_dY, vcl_dZ, vcl_X, vcl_Y, vcl_Z, vcl_R, sigma, b) );

    viennacl::ocl::get_queue().finish();
    if ( diff(dv0, vcl_dX) > epsilon
         || diff(dv1, vcl_dY) > epsilon
         || diff(dv2, vcl_dZ) > epsilon){
        std::cout << "# Error at operation: dX = sigma * (X - Y) ; ... " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( dv0, vcl_dX ) ) << std::endl;
        std::cout << "  diff_vec1: " << std::abs ( diff ( dv1, vcl_dY ) ) << std::endl;
        std::cout << "  diff_vec2: " << std::abs ( diff ( dv2, vcl_dZ ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();

    std::cout << "Testing dX -= beta * X * X * X;" << std::endl;
    
    NumericT beta = sigma;
    viennacl::copy(v0,vcl_X);
    viennacl::copy(dv0,vcl_dX);
    dv0 -= beta * boost::numeric::ublas::element_prod(v0, boost::numeric::ublas::element_prod(v0, v0));
    
    cpu_symbolic_scalar<2,NumericT> sym_beta;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symv0 -= sym_beta * element_prod(symv1, element_prod(symv1, symv1))
                                                                    ,"test_nested_element_prod") ( vcl_dX, vcl_X, beta) );


    viennacl::ocl::get_queue().finish();
    if ( diff(dv0, vcl_dX) > epsilon){
        std::cout << "# Error at operation: dX -= beta * X * X * X ; ... " << std::endl;
        std::cout << "  diff_vec0: " << std::abs ( diff ( dv0, vcl_dX ) ) << std::endl;
        retval = EXIT_FAILURE;
    }
    viennacl::ocl::get_queue().finish();
    
    return retval;

}

int main() {
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Test :: Multiline Operations" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    int retval = EXIT_SUCCESS;

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
    {
        typedef float NumericT;
        NumericT epsilon = 1.0E-3;
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  eps:     " << epsilon << std::endl;
        std::cout << "  numeric: float" << std::endl;

        std::cout << "---- Layout : Row Major" << std::endl;
        retval = test<NumericT, viennacl::row_major,ublas::row_major,1> ( epsilon );
        //retval = test<NumericT, viennacl::row_major,ublas::row_major,16> ( epsilon );

        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }

    return retval;
}
