/* =========================================================================
   Copyright (c) 2010-2012, Institute for Microelectronics,
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
#define VIENNACL_WITH_UBLAS 1

//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/reduce.hpp"
#include "viennacl/device_specific/code_generator.hpp"
#include "viennacl/scheduler/io.hpp"

#define CHECK_RESULT(cpu,gpu, op) \
    if ( double delta = fabs ( diff ( cpu, gpu) ) > epsilon ) {\
        std::cout << "# Error at operation: " #op << std::endl;\
        std::cout << "  diff: " << delta << std::endl;\
        retval = EXIT_FAILURE;\
    }\


using namespace boost::numeric;
using namespace viennacl;

template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(ublas::matrix<ScalarType> & mat1, VCLMatrixType & mat2)
{
   ublas::matrix<ScalarType> mat2_cpu(mat2.size1(), mat2.size2());
   viennacl::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   viennacl::copy(mat2, mat2_cpu);
   double ret = 0;
   double act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size1(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu.size2(); ++j)
      {
         act = std::fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max<ScalarType>( std::fabs(mat2_cpu(i, j)), std::fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}

template<typename ScalarType, unsigned int Alignment>
ScalarType diff ( ublas::vector<ScalarType> & v1, viennacl::vector<ScalarType,Alignment> & v2 ) {
    ublas::vector<ScalarType> v2_cpu ( v2.size() );
    viennacl::copy( v2.begin(), v2.end(), v2_cpu.begin() );
    for ( unsigned int i=0; i<v1.size(); ++i ) {
        if ( std::max ( std::fabs ( v2_cpu[i] ), std::fabs ( v1[i] ) ) > 0 )
            v2_cpu[i] = std::fabs ( v2_cpu[i] - v1[i] ) / std::max<ScalarType>( std::fabs ( v2_cpu[i] ), std::fabs ( v1[i] ) );
        else
            v2_cpu[i] = 0.0;
    }
    return norm_inf ( v2_cpu );
}


template< typename NumericT, class Layout, typename Epsilon >
int test( Epsilon const& epsilon) {
    int retval = EXIT_SUCCESS;

    ublas::vector<NumericT> cx;
    ublas::vector<NumericT> cy;

    ublas::matrix<NumericT> cA;
    ublas::matrix<NumericT> cB;
    ublas::matrix<NumericT> cC;
    ublas::matrix<NumericT> cD;

    unsigned int size1 = 762;
    unsigned int size2 = 663;

    cA.resize(size1,size2);
    cx.resize(size2);
    cy.resize(size1);

    srand(0);

    for (unsigned int i=0; i<size1; ++i){
        for (unsigned int j=0; j<size2; ++j){
            cA(i,j)=j;
        }
    }

    for (unsigned int i=0; i<size2; ++i){
        cx(i) = i;
    }

    for (unsigned int i=0; i<size1; ++i){
        cy(i) = i;
    }

//    std::cout << "Running tests for matrix of size " << cA.size1() << "," << cA.size2() << std::endl;

    viennacl::matrix<NumericT,Layout> A (size1, size2);
    viennacl::matrix<NumericT,Layout> B (size1, size2);
    viennacl::matrix<NumericT,Layout> C (size1, size2);
    viennacl::matrix<NumericT,Layout> D (size1, size2);

    viennacl::vector<NumericT> x(size2);
    viennacl::vector<NumericT> y(size1);


    cB = cA;
    cC = cA;
    cD = cA;
    viennacl::copy(cA,A);
    viennacl::copy(cB,B);
    viennacl::copy(cC,C);
    viennacl::copy(cD,D);

    viennacl::copy(cx,x);
    viennacl::copy(cy,y);


    // --------------------------------------------------------------------------
    {
        std::cout << "y = A*x..." << std::endl;
        cy     =  ublas::prod(cA,cx);
        viennacl::scheduler::statement statement(y, viennacl::op_assign(), viennacl::linalg::prod(A,x));
        //std::cout << statement << std::endl;
        device_specific::generate_enqueue_statement(statement, statement.array()[0]);
        viennacl::backend::finish();
        CHECK_RESULT(cy,y,y=A*x)
    }

    {
        std::cout << "x = trans(A)*y..." << std::endl;
        cx     =  ublas::prod(trans(cA),cy);
        viennacl::scheduler::statement statement(x, viennacl::op_assign(), viennacl::linalg::prod(trans(A),y));
        device_specific::generate_enqueue_statement(statement, statement.array()[0]);
        viennacl::backend::finish();
        CHECK_RESULT(cx,x,x=trans(A)*y)
    }

    {
        std::cout << "y = reduce_rows<add>(A)..." << std::endl;
        for (unsigned int i = 0; i < size1; ++i){
            NumericT acc = cA(i,0);
            for (unsigned int j = 1; j < size2; ++j){
                acc += cA(i,j);
            }
            cy(i) = acc;
        }
        viennacl::scheduler::statement statement(y, viennacl::op_assign(), viennacl::linalg::reduce_rows<viennacl::op_add>(A));
        //std::cout << statement << std::endl;

        device_specific::generate_enqueue_statement(statement, statement.array()[0]);
        viennacl::backend::finish();
        CHECK_RESULT(cy,y,y = reduce_rows<max>(A))
    }

    {
        std::cout << "x = reduce_columns<add>(A)..." << std::endl;
        for (unsigned int j = 0; j < size2; ++j){
            NumericT acc = cA(0,j);
            for (unsigned int i = 1; i < size1; ++i){
                acc += cA(i,j);
            }
            cx(j) = acc;
        }
        viennacl::scheduler::statement statement(x, viennacl::op_assign(), viennacl::linalg::reduce_columns<viennacl::op_add>(A));
        device_specific::generate_enqueue_statement(statement, statement.array()[0]);
        viennacl::backend::finish();
        CHECK_RESULT(cx,x,x = reduce_columns<max>(A))
    }


    return retval;
}


int main() {
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "## Test :: Generated BLAS2" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;

    int retval = EXIT_SUCCESS;

    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
    {
        double epsilon = 1.0E-4;
        std::cout << "# Testing setup:" << std::endl;
        std::cout << "  numeric: float" << std::endl;
        std::cout << "  --------------" << std::endl;
        std::cout << "  Row-Major"      << std::endl;
        std::cout << "  --------------" << std::endl;
        retval = test<float, viennacl::row_major> (epsilon);
        std::cout << "  --------------" << std::endl;
        std::cout << "  Column-Major"   << std::endl;
        std::cout << "  --------------" << std::endl;
        retval &= test<float, viennacl::column_major> (epsilon);

        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }

//    std::cout << std::endl;
//    std::cout << "----------------------------------------------" << std::endl;
//    std::cout << std::endl;
//#ifdef VIENNACL_WITH_OPENCL
//   if ( viennacl::ocl::current_device().double_support() )
//#endif
//    {
//        double epsilon = 1.0E-4;
//        std::cout << "# Testing setup:" << std::endl;
//        std::cout << "  numeric: double" << std::endl;
//        std::cout << "  --------------" << std::endl;
//        std::cout << "  Row-Major"      << std::endl;
//        std::cout << "  --------------" << std::endl;
//        retval = test<double, viennacl::row_major> (epsilon);
//        std::cout << "  --------------" << std::endl;
//        std::cout << "  Column-Major"   << std::endl;
//        std::cout << "  --------------" << std::endl;
//        retval &= test<double, viennacl::column_major> (epsilon);

//        if ( retval == EXIT_SUCCESS )
//            std::cout << "# Test passed" << std::endl;
//        else
//            return retval;
//    }
}
