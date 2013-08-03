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

//#define VIENNACL_DEBUG_ALL
//#define VIENNACL_DEBUG_BUILD

//
// include necessary system headers
//
#include <iostream>
#include <iomanip>

//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

#include "viennacl/generator/generate.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

#define N_RUNS 100
#define SIZE_INC 256
#define MAX_SIZE 7936

template<typename ScalarType, class FB>
double run_benchmark(size_t size, bool is_trans)
{
    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::vector<ScalarType> y = viennacl::scalar_vector<ScalarType>(size,1);
    viennacl::matrix<ScalarType,FB> A = viennacl::scalar_matrix<ScalarType>(size, size,1);
    viennacl::vector<ScalarType> x = viennacl::scalar_vector<ScalarType>(size,1);

    viennacl::scheduler::statement * statement;

    if(is_trans)
      statement = new viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(trans(A),x));
    else
      statement = new viennacl::scheduler::statement(y, viennacl::op_assign(), viennacl::linalg::prod(A,x));

    viennacl::generator::generate_enqueue_statement(*statement, statement->array()[0]);
    viennacl::backend::finish();

    Timer timer;
    timer.start();
    for(unsigned int r = 0 ; r < N_RUNS ; ++r){
      viennacl::generator::generate_enqueue_statement(*statement, statement->array()[0]);
    }
    viennacl::backend::finish();

    double time = timer.get()/(double)N_RUNS;
    delete statement;
    return 1e-9*size*(2*size-1)/time;
}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argv,argv+argc);

    typedef std::vector< viennacl::ocl::platform > platforms_type;
    typedef std::vector<viennacl::ocl::device> devices_type;
    typedef std::vector<cl_device_id> cl_devices_type;

    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();

    std::cout << "Running GEMV..." << std::endl;
    for(unsigned int k=0 ; k < num_platforms ; ++k)
    {
        viennacl::ocl::platform pf(k);
        viennacl::ocl::set_context_device_type(k,CL_DEVICE_TYPE_ALL);
        viennacl::ocl::set_context_platform_index(k,k);
        viennacl::ocl::switch_context(k);
        devices_type dev = viennacl::ocl::current_context().devices();
        for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){
                viennacl::ocl::switch_device(*it);
                std::cout << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << "               Device Info" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << viennacl::ocl::current_device().info() << std::endl;
                std::cout << std::endl;
                std::cout << "float:" << std::endl;
                std::cout << "#N\tAv(GFLOP/s)\tTv(GFLOP/s)" << std::endl;
                for(unsigned int size = SIZE_INC ; size <= MAX_SIZE ; size += SIZE_INC){
                    std::cout << size << "\t" << std::setprecision(3) << run_benchmark<float,viennacl::row_major>(size,false) << "\t" << run_benchmark<float,viennacl::row_major>(size,true) << std::endl;
                }
                std::cout << std::endl;
                std::cout << "double:" << std::endl;
                std::cout << "#N\tAv(GFLOP/s)\tTv(GFLOP/s)" << std::endl;
                for(unsigned int size = SIZE_INC ; size <= MAX_SIZE ; size += SIZE_INC){
                  std::cout << size << "\t" << std::setprecision(3) << run_benchmark<double,viennacl::row_major>(size,false) << "\t" << run_benchmark<double,viennacl::row_major>(size,true) << std::endl;
                }
        }
    }
    return 0;
}
