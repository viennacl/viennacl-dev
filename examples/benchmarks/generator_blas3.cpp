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

/*
*
*   Benchmark: BLAS level 3 functionality for dense matrices (blas3.cpp and blas3.cu are identical, the latter being required for compilation using CUDA nvcc)
*
*/

//disable debug mechanisms to have a fair benchmark environment
#ifndef NDEBUG
#define NDEBUG
#endif

#define VIENNACL_DEBUG_BUILD

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/lu.hpp"

#include "viennacl/generator/custom_operation.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

#define N_RUNS 5
#define SIZE_INC 256
#define MAX_SIZE 7936

template<typename ScalarType, class FB, class FC>
double run_benchmark(size_t matrix_size)
{

    //
    // One alternative: Put the matrices into a contiguous block of memory (allows to use viennacl::fast_copy(), avoiding temporary memory)
    //
    std::vector<ScalarType> stl_B(matrix_size * matrix_size);
    std::vector<ScalarType> stl_C(matrix_size * matrix_size);

    //
    // Fill the matrix
    //
    for (unsigned int i = 0; i < matrix_size; ++i)
        for (unsigned int j = 0; j < matrix_size; ++j)
            stl_B[i*matrix_size + j] = random<ScalarType>();

    for (unsigned int i = 0; i < matrix_size; ++i)
        for (unsigned int j = 0; j < matrix_size; ++j)
            stl_C[i + j*matrix_size] = random<ScalarType>();



    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::matrix<ScalarType> vcl_A(matrix_size, matrix_size);
    viennacl::matrix<ScalarType,FB> vcl_B(matrix_size, matrix_size);
    viennacl::matrix<ScalarType,FC> vcl_C(matrix_size, matrix_size);

    typedef viennacl::generator::matrix< viennacl::matrix<ScalarType> > dma_t;
    typedef viennacl::generator::matrix< viennacl::matrix<ScalarType,FB> > dmb_t;
    typedef viennacl::generator::matrix< viennacl::matrix<ScalarType,FC> > dmc_t;

    viennacl::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        vcl_B);
    viennacl::fast_copy(&(stl_C[0]),
                        &(stl_C[0]) + stl_C.size(),
                        vcl_C);

    viennacl::generator::custom_operation op;
    op.add(dma_t(vcl_A) = viennacl::generator::prod(dmb_t(vcl_B), dmc_t(vcl_C)));
    op.program();
    op.execute();
    viennacl::backend::finish();

    double res = 0;
    Timer timer;
    timer.start();
    for(unsigned int r = 0 ; r < N_RUNS ; ++r){
        op.execute();
    }
    viennacl::backend::finish();
    res = timer.get();

    return res/N_RUNS;
}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argv,argv+argc);
    if(argc<3){
        std::cerr << "USAGE : PROGRAM_NAME DEVICE LAYOUT SCALARTYPE" << std::endl;
        exit(1);
    }

    unsigned int requested_device = atoi(args[1].c_str());
    unsigned int layout = atoi(args[2].c_str());
    std::string scalartype = args[3];

    typedef std::vector< viennacl::ocl::platform > platforms_type;
    typedef std::vector<viennacl::ocl::device> devices_type;
    typedef std::vector<cl_device_id> cl_devices_type;

    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();

    unsigned int current_device = 0;

    for(unsigned int k=0 ; k < num_platforms ; ++k)
    {
        viennacl::ocl::platform pf(k);
        viennacl::ocl::set_context_device_type(k,CL_DEVICE_TYPE_ALL);
        viennacl::ocl::set_context_platform_index(k,k);
        viennacl::ocl::switch_context(k);
        devices_type dev = viennacl::ocl::current_context().devices();
        for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){

            if(current_device++ == requested_device ){
                viennacl::ocl::switch_device(*it);
                std::cout << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << "               Device Info" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << viennacl::ocl::current_device().info() << std::endl;
                std::cout << "Running GEMM for : " << scalartype << std::endl;

                std::cout << "#Size \t Execution Time" << std::endl;
                for(unsigned int size = SIZE_INC ; size <= MAX_SIZE ; size += SIZE_INC){
                    double exec_time = 0;
                    if(scalartype=="float"){
                        switch(layout){
                        case 0 : exec_time = run_benchmark<float,viennacl::row_major,viennacl::row_major>(size); break;
                        case 1 : exec_time = run_benchmark<float,viennacl::column_major,viennacl::row_major>(size); break;
                        case 2 : exec_time = run_benchmark<float,viennacl::row_major,viennacl::column_major>(size); break;
                        case 3 : exec_time = run_benchmark<float,viennacl::column_major,viennacl::column_major>(size); break;
                        }
                    }
                    else if(scalartype=="double"){
                        switch(layout){
                        case 0 : exec_time = run_benchmark<double,viennacl::row_major,viennacl::row_major>(size); break;
                        case 1 : exec_time = run_benchmark<double,viennacl::column_major,viennacl::row_major>(size); break;
                        case 2 : exec_time = run_benchmark<double,viennacl::row_major,viennacl::column_major>(size); break;
                        case 3 : exec_time = run_benchmark<double,viennacl::column_major,viennacl::column_major>(size); break;
                        }
                    }
                    else{
                        std::cerr << "Unknown Scalartype ... Aborting" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    std::cout << size << "\t" << exec_time << std::endl;
                }
            }
        }
    }
    return 0;
}
