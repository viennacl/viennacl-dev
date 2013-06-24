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

//
// ViennaCL includes
//
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

#include "viennacl/generator/custom_operation.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

#define N_RUNS 100
#define MAX_SIZE 1e8

template<typename ScalarType>
float run_benchmark(size_t size)
{
    float res;
    viennacl::vector<ScalarType> vcl_A(size);
    viennacl::vector<ScalarType> vcl_B(size);

    viennacl::scalar<ScalarType> s(0);

    typedef viennacl::generator::vector< ScalarType > vec;
    typedef viennacl::generator::scalar< ScalarType > scal;

    std::vector<ScalarType> stl_A(size);
    std::vector<ScalarType> stl_B(size);


    for (unsigned int i = 0; i < size; ++i) stl_A[i] = random<ScalarType>();
    for (unsigned int i = 0; i < size; ++i)  stl_B[i] = random<ScalarType>();

    viennacl::fast_copy(stl_A,vcl_A);
    viennacl::fast_copy(stl_B,vcl_B);


    viennacl::generator::custom_operation inprod;
    inprod.add(scal(s) = inner_prod(vec(vcl_A),2*vec(vcl_A)+vec(vcl_B)));
    inprod.execute();
    viennacl::backend::finish();

    Timer timer;
    timer.start();
    for(unsigned int r = 0 ; r < N_RUNS ; ++r){
        inprod.execute();
    }
    viennacl::backend::finish();
    res = timer.get()/(double)N_RUNS;
    return res;
}

int main(int argc, char* argv[]){
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
                std::cout << "#Size \t Time" << std::endl;
                for(unsigned int size = 1024 ; size <= MAX_SIZE ; size *= 2){
                     float exec_time = run_benchmark<float>(size);
                    std::cout << size << "\t" << exec_time << std::endl;
                }
                std::cout << std::endl;
                std::cout << "double:" << std::endl;
                std::cout << "#Size \t Time" << std::endl;
                for(unsigned int size = 1024 ; size <= MAX_SIZE ; size *= 2){
                     float exec_time = run_benchmark<double>(size);
                    std::cout << size << "\t" << exec_time << std::endl;
                }
        }
    }
    return 0;
}
