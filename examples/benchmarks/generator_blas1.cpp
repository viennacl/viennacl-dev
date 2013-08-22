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

//#define VIENNACL_DEBUG_BUILD

//
// include necessary system headers
//
#include <iostream>
#include <iomanip>

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#include "viennacl/generator/generate.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"

#include "viennacl/scheduler/forwards.h"

#include "benchmark-utils.hpp"

#define N_RUNS 100
#define MAX_SIZE 1e8

enum operation_type{
  dot,
  assign
};

template<typename ScalarType>
float run_benchmark(size_t size, operation_type type)
{
    std::size_t n_bytes = size*sizeof(ScalarType);
    std::size_t n_transfers;
    if(type==dot)
      n_transfers = 2;
    else if(type==assign)
      n_transfers = 2;
    viennacl::vector<ScalarType> vcl_A = viennacl::scalar_vector<ScalarType>(size,1);
    viennacl::vector<ScalarType> vcl_B = viennacl::scalar_vector<ScalarType>(size,1);

    viennacl::scalar<ScalarType> s(0);

    viennacl::scheduler::statement * statement;

    if(type==dot)
      statement = new viennacl::scheduler::statement(s, viennacl::op_assign(), viennacl::linalg::inner_prod(vcl_A, vcl_B));
    else if(type==assign)
      statement = new viennacl::scheduler::statement(vcl_A, viennacl::op_assign(), vcl_B);

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

    return (n_bytes*n_transfers)/time / 1e9;
}


int main(){
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    typedef std::vector<viennacl::ocl::device> devices_type;
    typedef std::vector<cl_device_id> cl_devices_type;

    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();

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
                std::cout << "float:" << std::endl;
                std::cout << "#N\tAssign(GB/s)\tDot(GB/s)" << std::endl;
                for(unsigned int size = 1024 ; size <= MAX_SIZE ; size *= 2){
                  std::cout << std::setprecision(2) << (float)size << "\t" << (int)run_benchmark<float>(size, assign) << "\t" << (int)run_benchmark<float>(size, dot) << std::endl;
                }
                std::cout << std::endl;
                std::cout << "double:" << std::endl;
                std::cout << "#N\tAssign(GB/s)\tDot(GB/s)" << std::endl;
                for(unsigned int size = 1024 ; size <= MAX_SIZE ; size *= 2){
                  std::cout << std::setprecision(2) << (double)size << "\t" << (int)run_benchmark<double>(size, assign) << "\t" << (int)run_benchmark<double>(size, dot) << std::endl;
                }
        }
    }
    return 0;
}
