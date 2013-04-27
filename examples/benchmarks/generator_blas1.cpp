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
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/lu.hpp"

#include "viennacl/generator/custom_operation.hpp"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

#define N_RUNS 5
#define MAX_SIZE 1e7

template<typename ScalarType>
std::pair<double, double> run_benchmark(size_t size)
{
    std::pair<double, double> res;
    viennacl::vector<ScalarType> vcl_A(size);
    viennacl::vector<ScalarType> vcl_B(size);
    viennacl::vector<ScalarType> vcl_C(size);

    viennacl::scalar<ScalarType> s(0);

    typedef viennacl::generator::vector< ScalarType > vec;
    typedef viennacl::generator::scalar< ScalarType > scal;

    std::vector<ScalarType> stl_B(size);
    std::vector<ScalarType> stl_C(size);

    for (unsigned int i = 0; i < size; ++i) stl_B[i] = random<ScalarType>();
    for (unsigned int i = 0; i < size; ++i)  stl_C[i] = random<ScalarType>();

    viennacl::fast_copy(stl_B,vcl_B);
    viennacl::fast_copy(stl_C,vcl_C);

    //SAXPY
    {
        viennacl::generator::custom_operation saxpy;
        saxpy.add(vec(vcl_A) = vec(vcl_B) + vec(vcl_C));
        saxpy.execute();
        viennacl::backend::finish();

        double time = 0;
        for(unsigned int r = 0 ; r < N_RUNS ; ++r){
            Timer timer;
            timer.start();
            saxpy.execute();
            viennacl::backend::finish();
            time += timer.get();
        }
        res.first = time/(double)N_RUNS;
    }

    //INNER PRODUCT
    {
        viennacl::generator::custom_operation inprod;
        inprod.add(scal(s) = inner_prod(vec(vcl_B),vec(vcl_C)));
        inprod.execute();
        viennacl::backend::finish();

        double time = 0;
        for(unsigned int r = 0 ; r < N_RUNS ; ++r){
            Timer timer;
            timer.start();
            inprod.execute();
            viennacl::backend::finish();
            time += timer.get();
        }
        res.second = time/(double)N_RUNS;
    }

    return res;
}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argv,argv+argc);
    if(argc!=3){
        std::cerr << "USAGE : PROGRAM_NAME DEVICE SCALARTYPE" << std::endl;
        exit(1);
    }

    unsigned int requested_device = atoi(args[1].c_str());
    std::string scalartype = args[2];

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
                std::cout << "Running GEMV for : " << scalartype << std::endl;

                std::cout << "#Size \t SAXPY  \t Inner Product" << std::endl;
                for(unsigned int size = 1024 ; size <= MAX_SIZE ; size *= 2){
                    std::pair<double,double> exec_time;
                    if(scalartype=="float"){
                        exec_time = run_benchmark<float>(size);
                    }
                    else if(scalartype=="double"){
                        exec_time = run_benchmark<double>(size);
                    }
                    else{
                        std::cerr << "Unknown Scalartype ... Aborting" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    std::cout << size << "\t" << exec_time.first << "\t" << exec_time.second << " #"  << std::endl;
                }
            }
        }
    }
    return 0;
}
