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
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_HAVE_UBLAS 1
#include "viennacl/scalar.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "examples/tutorial/Random.hpp"
#include "examples/benchmarks/benchmark-utils.hpp"
#include "viennacl/generator/custom_operation.hpp"

using namespace boost::numeric;

const int matrix_size = 100;

template <typename ScalarType, typename F, unsigned int ALIGNMENT>
ScalarType diff ( ublas::matrix<ScalarType> & mat1, viennacl::matrix<ScalarType, F, ALIGNMENT> & mat2 ) {
    ublas::matrix<ScalarType> mat2_cpu ( mat2.size1(), mat2.size2() );
    copy ( mat2, mat2_cpu );
    ScalarType ret = 0;
    ScalarType act = 0;

    for ( unsigned int i = 0; i < mat2_cpu.size1(); ++i ) {
        for ( unsigned int j = 0; j < mat2_cpu.size2(); ++j ) {
            act = fabs ( mat2_cpu ( i,j ) - mat1 ( i,j ) ) / std::max ( fabs ( mat2_cpu ( i, j ) ), fabs ( mat1 ( i,j ) ) );
            if ( act > ret )
                ret = act;
        }
    }
    //std::cout << ret << std::endl;
    return ret;
}

template< typename NumericT, typename Epsilon >
int test ( Epsilon const& epsilon ) {

    int retval = EXIT_SUCCESS;

    ublas::matrix<NumericT> mat ( matrix_size, matrix_size );

    NumericT                    cpu_scal = static_cast<NumericT> ( 42.1415 );
    viennacl::scalar<NumericT>  gpu_scal = static_cast<NumericT> ( 42.1415 );

    viennacl::matrix<NumericT> vcl_mat ( matrix_size, matrix_size );
    viennacl::matrix<NumericT> vcl_mat2 ( matrix_size, matrix_size );

    viennacl::generator::symbolic_matrix<0,NumericT> symm;
    viennacl::generator::symbolic_matrix<1,NumericT> symm2;

    viennacl::generator::cpu_symbolic_scalar<1,NumericT> cpu_sym_scal2;

    viennacl::generator::gpu_symbolic_scalar<1,NumericT> gpu_sym_scal2;

    for ( unsigned int i = 0; i < mat.size1(); ++i )
        for ( unsigned int j = 0; j < mat.size2(); ++j )
            mat ( i,j ) = static_cast<NumericT> ( 0.1 ) * random<NumericT>();

    ublas::matrix<NumericT> mat2 ( mat ) ;

    viennacl::copy ( mat, vcl_mat );
    viennacl::copy ( mat2, vcl_mat2 );

    std::cout << "Testing addition..." << std::endl;
    mat     = mat + mat2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm + symm2 ) ( vcl_mat, vcl_mat2 ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "Testing inplace addition..." << std::endl;
    mat     += mat2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm += symm2 ) ( vcl_mat, vcl_mat2 ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing substraction..." << std::endl;
    mat     = mat - mat2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm - symm2 ) ( vcl_mat, vcl_mat2 ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: substraction" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "Testing inplace substraction..." << std::endl;
    mat     -= mat2;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm -= symm2 ) ( vcl_mat, vcl_mat2 ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace addition" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    // --------------------------------------------------------------------------

    std::cout << "testing cpu scalar multiplication ..." << std::endl;
    mat     = mat*cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm*cpu_sym_scal2 ) ( vcl_mat, cpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: cpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace cpu scalar multiplication ..." << std::endl;
    mat     *= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm *= cpu_sym_scal2 ) ( vcl_mat, cpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace cpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing cpu scalar division ..." << std::endl;
    mat     = mat/cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm/cpu_sym_scal2 ) ( vcl_mat, cpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: cpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace cpu scalar division ..." << std::endl;
    mat     /= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm /= cpu_sym_scal2 ) ( vcl_mat, cpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace cpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing gpu scalar multiplication ..." << std::endl;
    mat     = mat*cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm*gpu_sym_scal2 ) ( vcl_mat, gpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: gpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace gpu scalar multiplication ..." << std::endl;
    mat     *= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm *= gpu_sym_scal2 ) ( vcl_mat, gpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace gpu scalar multiplication " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing gpu scalar division ..." << std::endl;
    mat     = mat/cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm = symm/gpu_sym_scal2 ) ( vcl_mat, gpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: gpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    std::cout << "testing inplace gpu scalar division ..." << std::endl;
    mat     /= cpu_scal;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation ( symm /= gpu_sym_scal2 ) ( vcl_mat, gpu_scal ) );
    if ( fabs ( diff ( mat, vcl_mat ) ) > epsilon ) {
        std::cout << "# Error at operation: inplace gpu scalar division " << std::endl;
        std::cout << "  diff: " << fabs ( diff ( mat, vcl_mat ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

    return retval;

}

int main() {
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Test :: Matrix" << std::endl;
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
        std::cout << "  layout: row-major" << std::endl;
        retval = test<NumericT> ( epsilon );
        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }
}
