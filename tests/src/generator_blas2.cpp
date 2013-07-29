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
#include <boost/foreach.hpp>

//
// *** ViennaCL
//
#define VIENNACL_WITH_UBLAS 1

//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

#include "viennacl/linalg/prod.hpp"

#include "viennacl/generator/generate.hpp"

#define CHECK_RESULT(cpu,gpu, op) \
    if ( double delta = fabs ( diff ( cpu, gpu) ) > epsilon ) {\
        std::cout << "# Error at operation: " #op << std::endl;\
        std::cout << "  diff: " << delta << std::endl;\
        retval = EXIT_FAILURE;\
    }\


using namespace boost::numeric;
using namespace viennacl;

template <typename ScalarType, typename VCLMatrixType>
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
         act = fabs(mat2_cpu(i,j) - mat1(i,j)) / std::max( fabs(mat2_cpu(i, j)), fabs(mat1(i,j)) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
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


template< typename NumericT, class Layout, typename Epsilon >
int test( Epsilon const& epsilon) {
    int retval = EXIT_SUCCESS;

    ublas::vector<NumericT> cx;
    ublas::vector<NumericT> cy;

    ublas::matrix<NumericT> cA;
    ublas::matrix<NumericT> cB;
    ublas::matrix<NumericT> cC;
    ublas::matrix<NumericT> cD;

    unsigned int size1 = 841;
    unsigned int size2 = 772;

    cA.resize(size1,size2);
    cx.resize(size2);
    cy.resize(size1);

    for(unsigned int i=0; i<size1; ++i){
        for(unsigned int j=0 ; j<size2; ++j){
            cA(i,j)=(double)std::rand()/RAND_MAX;
        }
    }

    for(unsigned int i=0; i<size2; ++i){
        cx(i) = (double)std::rand()/RAND_MAX;
    }

    for(unsigned int i=0; i<size1; ++i){
        cy(i) = (double)std::rand()/RAND_MAX;
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
        generator::code_generator generator;
        generator.add(viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(A,x)));
        generator::enqueue(generator);
        viennacl::backend::finish();
        CHECK_RESULT(cy,y,y=A*x)
    }

    {
        std::cout << "x = trans(A)*y..." << std::endl;
        cy     =  ublas::prod(trans(cA),cx);
        generator::code_generator generator;
        generator.add(viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(trans(A),x)));
        generator::enqueue(generator);
        viennacl::backend::finish();
        CHECK_RESULT(cx,x,x=trans(A)*y)
    }

//    {
//        std::cout << "y = reduce_rows<max>(A)..." << std::endl;
//        for(unsigned int i = 0 ; i < size1 ; ++i){
//            NumericT current_max = -INFINITY;
//            for(unsigned int j = 0 ; j < size2 ; ++j){
//                current_max = std::max(current_max,cA(i,j));
//            }
//            cy(i) = current_max;
//        }
//        generator::custom_operation op;
//        op.add(dv_t(y) = generator::reduce_rows<generator::fmax_type>(dm_t(A)));
//        op.execute();
//        viennacl::backend::finish();
//        CHECK_RESULT(cy,y,y = reduce_rows<max>(A))
//    }


//    {
//        std::cout << "x = reduce_cols<max>(A)..." << std::endl;
//        for(unsigned int j = 0 ; j < size2 ; ++j){
//            NumericT current_max = -INFINITY;
//            for(unsigned int i = 0 ; i < size1 ; ++i){
//                current_max = std::max(current_max,cA(i,j));
//            }
//            cx(j) = current_max;
//        }
//        generator::custom_operation op;
//        op.add(dv_t(x) = generator::reduce_cols<generator::fmax_type>(dm_t(A)));
//        op.execute();
//        viennacl::backend::finish();
//        CHECK_RESULT(cx,x,x = reduce_cols<max>(A))
//    }


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
        std::cout << "  numeric: double" << std::endl;
        std::cout << "  --------------" << std::endl;
        std::cout << "  Row-Major"      << std::endl;
        std::cout << "  --------------" << std::endl;
        retval = test<double, viennacl::row_major> (epsilon);
        std::cout << "  --------------" << std::endl;
        std::cout << "  Column-Major"   << std::endl;
        std::cout << "  --------------" << std::endl;
        retval &= test<double, viennacl::column_major> (epsilon);

        if ( retval == EXIT_SUCCESS )
            std::cout << "# Test passed" << std::endl;
        else
            return retval;
    }
}
