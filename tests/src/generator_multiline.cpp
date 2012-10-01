
 #define VIENNACL_DEBUG_CUSTOM_OPERATION
#define VIENNACL_DEBUG_BUILD

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
// #define VIENNACL_HAVE_UBLAS 1
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
#include "viennacl/generator/symbolic_types.hpp"

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
        if ( std::max ( fabs ( v2_cpu[i] ), fabs ( v1[i] ) ) > 0 )
            v2_cpu[i] = fabs ( v2_cpu[i] - v1[i] ) / std::max ( fabs ( v2_cpu[i] ), fabs ( v1[i] ) );
        else
            v2_cpu[i] = 0.0;
    }
    return norm_inf ( v2_cpu );
}

template< typename NumericT,  typename F,typename F2, unsigned int Alignment, typename Epsilon >
int test ( Epsilon const& epsilon ) {
    int retval = EXIT_SUCCESS;
    static const unsigned int SIZE = 10;
    // --------------------------------------------------------------------------
    NumericT scalar;
    ublas::vector<NumericT> rhs ( SIZE );
    for ( unsigned int i = 0; i < rhs.size(); ++i )
        rhs ( i ) = i;
    ublas::vector<NumericT> rhs2 = rhs;
    ublas::vector<NumericT> result = ublas::scalar_vector<NumericT> ( SIZE, 1 );
    ublas::vector<NumericT> result2 = result;
    ublas::vector<NumericT> rhs_trans = rhs;
    rhs_trans.resize ( result.size(), true );
    ublas::vector<NumericT> result_trans = ublas::zero_vector<NumericT> ( rhs.size() );



    ublas::matrix<NumericT,F2> matrix ( result.size(), rhs.size() );
    for ( unsigned int i = 0; i < matrix.size1(); ++i )
        for ( unsigned int j = 0; j < matrix.size2(); ++j )
            matrix ( i,j ) = i+j;


    std::cout << "----- Alignment " << Alignment << " -----" << std::endl;

    viennacl::scalar<NumericT> vcl_scalar;
    viennacl::vector<NumericT,Alignment> vcl_rhs ( rhs.size() );
    viennacl::vector<NumericT,Alignment> vcl_rhs_trans ( rhs_trans.size() );
    viennacl::vector<NumericT,Alignment> vcl_result_trans ( result_trans.size() );
    viennacl::vector<NumericT,Alignment> vcl_result ( result.size() );
    viennacl::matrix<NumericT, F, Alignment> vcl_matrix ( rhs.size(), rhs.size() );

    viennacl::copy ( rhs.begin(), rhs.end(), vcl_rhs.begin() );
    viennacl::copy ( result, vcl_result );
    viennacl::copy ( matrix, vcl_matrix );

    // --------------------------------------------------------------------------

    symbolic_vector<0,float> symv0;
    symbolic_matrix<1,float> symm1;
    symbolic_vector<2,float> symv2;
    gpu_symbolic_scalar<3,float> syms3;
    // --------------------------------------------------------------------------
    std::cout << "matrix-vector product" << std::endl;
    result     = ublas::prod ( matrix, rhs );
    scalar = ublas::inner_prod(rhs, rhs);
    rhs = scalar*result;
    viennacl::ocl::enqueue ( viennacl::generator::custom_operation (symv0 = prod ( symm1,symv2 ),
                                                                    syms3 = inner_prod(symv2,symv2),
                                                                    symv2 = syms3 * symv0,
                                                                     "test") ( vcl_result,vcl_matrix,vcl_rhs,vcl_scalar ) );
    viennacl::ocl::get_queue().finish();

    if ( fabs ( diff ( result, vcl_result ) ) > epsilon ) {
        std::cout << "# Error at operation: matrix-vector product" << std::endl;
        std::cout << "  diff: " << fabs ( diff ( result, vcl_result ) ) << std::endl;
        retval = EXIT_FAILURE;
    }

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

        std::cout << "---- Layout : Column Major" << std::endl;
        retval = test<NumericT, viennacl::column_major,ublas::column_major,1> ( epsilon );

        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }

    return retval;
}
