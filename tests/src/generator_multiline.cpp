
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
#include "../examples/benchmarks/benchmark-utils.hpp"

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
    static const unsigned int SIZE = 10000;
    // --------------------------------------------------------------------------
    NumericT scalar;
    ublas::vector<NumericT> v0 ( SIZE );
    for ( unsigned int i = 0; i < v0.size(); ++i )
        v0 ( i ) = i;
    ublas::vector<NumericT> v1 = v0;
    ublas::vector<NumericT> v2 = v0;
    ublas::vector<NumericT> v3 = v0;
    ublas::matrix<NumericT,F2> matrix ( v0.size(), v1.size() );
    for ( unsigned int i = 0; i < matrix.size1(); ++i )
        for ( unsigned int j = 0; j < matrix.size2(); ++j )
            matrix ( i,j ) = i+j;


    std::cout << "----- Alignment " << Alignment << " -----" << std::endl;

    viennacl::scalar<NumericT> vcl_scalar(scalar);
    viennacl::vector<NumericT,Alignment> vcl_v0 ( v0.size() );
    viennacl::vector<NumericT,Alignment> vcl_v1 ( v1.size() );
    viennacl::vector<NumericT,Alignment> vcl_v2 ( v2.size() );
    viennacl::vector<NumericT,Alignment> vcl_v3 ( v3.size() );
    viennacl::matrix<NumericT, F, Alignment> vcl_matrix ( matrix.size1(), matrix.size2() );

    viennacl::copy ( v0, vcl_v0 );
    viennacl::copy ( v1, vcl_v1 );
    viennacl::copy ( v2, vcl_v2 );
    viennacl::copy ( v3, vcl_v3 );
    viennacl::copy ( matrix, vcl_matrix);

    // --------------------------------------------------------------------------

    symbolic_vector<0,float,Alignment> symv0;
    symbolic_vector<1,float,Alignment> symv1;
    symbolic_vector<2,float,Alignment> symv2;
    symbolic_vector<3,float,Alignment> symv3;

    gpu_symbolic_scalar<4,float> syms4;
    cpu_symbolic_scalar<5,float> sym_bound;
    // --------------------------------------------------------------------------

    std::cout << "Testing : Repeater..." << std::endl;

    float bound = 1;

    v0 = v1 + v2;
    for(unsigned int i=0 ; i<bound ; ++i){
        v2 = scalar*v0;
        for(unsigned int j=0 ; j<bound ; ++j){
            v3 = v0 - v1;
        }
    }
    v2 = v1 + v3;


    viennacl::generator::custom_operation op(symv0 = symv1 + symv2
                                                                        ,viennacl::generator::repeat(sym_bound,symv2 = syms4 * symv0
                                                                                                              , viennacl::generator::repeat(sym_bound,symv3 = symv0 - symv1))
                                                                        ,symv2 = symv1 + symv3
                                                                        ,"test");
    std::cout << op.kernels_source_code() << std::endl;
    viennacl::ocl::enqueue ( op ( vcl_v0, vcl_v1, vcl_v2, vcl_v3, vcl_scalar,bound) );
    viennacl::ocl::get_queue().finish();


    if ( fabs ( diff ( v0, vcl_v0) ) > epsilon
         || fabs ( diff (v2, vcl_v2) ) > epsilon
         || fabs ( diff(v3, vcl_v3) ) > epsilon ) {
        std::cout << "# Error at operation: Repeater" << std::endl;
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
        retval = test<NumericT, viennacl::row_major,ublas::row_major,16> ( epsilon );


        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }

    return retval;
}
