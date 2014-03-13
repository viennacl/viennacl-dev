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

//
// include necessary system headers
//
#include <iostream>

//
// ViennaCL includes
//
//#define VIENNACL_DEBUG_BUILD

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/lu.hpp"

#include "viennacl/generator/generate.hpp"
#include "viennacl/scheduler/forwards.h"

// Some helper functions for this tutorial:
#include "../tutorial/Random.hpp"


#include "benchmark-utils.hpp"

#define N_RUNS 5

static const std::size_t size_min = 128;
static const std::size_t size_max = 5120;
static const std::size_t size_inc = 128;

template<class MatA, class MatB, class MatC>
viennacl::scheduler::statement * allocate_statement(char ATrans, char BTrans, MatA const & A, MatB const & B, MatC const & C){
    if(ATrans=='N' && BTrans=='N')
        return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,B));
    else if(ATrans=='N' && BTrans=='T')
        return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(A,trans(B)));
    else if(ATrans=='T' && BTrans=='N')
        return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),B));
    else if(ATrans=='T' && BTrans=='T')
        return new viennacl::scheduler::statement(C, viennacl::op_assign(), viennacl::linalg::prod(trans(A),trans(B)));

}

template<typename ScalarType>
unsigned int run_benchmark(size_t size, char ATrans, char BTrans)
{    //viennacl::ocl::current_context().build_options("-cl-mad-enable -cl-fast-relaxed-math");   //uncomment for additional optimizations
    //viennacl::ocl::current_context().build_options("-cl-opt-disable");                        //uncomment to get poor performance
    viennacl::matrix<ScalarType, viennacl::column_major> A(size, size);
    viennacl::matrix<ScalarType, viennacl::column_major> B(size, size);
    viennacl::matrix<ScalarType, viennacl::column_major> C(size, size);
    viennacl::scheduler::statement * statement = allocate_statement(ATrans, BTrans,A,B,C);
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
    return static_cast<unsigned int>(2*pow(size/static_cast<double>(1000.0),3)/time);
}

int main(){
    typedef std::vector< viennacl::ocl::platform > platforms_type;
    typedef std::vector<viennacl::ocl::device> devices_type;

    platforms_type platforms = viennacl::ocl::get_platforms();
    size_t num_platforms = platforms.size();


    for(unsigned int k=0 ; k < num_platforms ; ++k)
    {
        viennacl::ocl::platform pf(k);
        viennacl::ocl::set_context_platform_index(k,k);
        viennacl::ocl::switch_context(k);
        devices_type dev = viennacl::ocl::current_context().devices();
        for(devices_type::iterator it = dev.begin() ; it != dev.end() ; ++it){
          if(it->type()==CL_DEVICE_TYPE_GPU){
                viennacl::ocl::switch_device(*it);
                std::cout << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << "               Device Info" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << viennacl::ocl::current_device().info() << std::endl;
                std::cout << "----------------------------------------------" << std::endl;

                std::cout << "float : " << std::endl;
                std::cout << "#Size\tNN\tNT\tTN\tTT" << std::endl;
                for(unsigned int size = size_min ; size <= size_max ; size += size_inc){
                    std::cout << size << "\t" << run_benchmark<float>(size,'N','N')
                              << "\t" << run_benchmark<float>(size,'N','T')
                              << "\t" << run_benchmark<float>(size,'T','N')
                              << "\t" << run_benchmark<float>(size,'T','T') << std::endl;
                }

                std::cout << "double : " << std::endl;
                std::cout << "#Size\tNN\tNT\tTN\tTT" << std::endl;
                for(unsigned int size = size_min ; size <= size_max ; size += size_inc){
                    std::cout << size << "\t" << run_benchmark<double>(size,'N','N')
                              << "\t" << run_benchmark<double>(size,'N','T')
                              << "\t" << run_benchmark<double>(size,'T','N')
                              << "\t" << run_benchmark<double>(size,'T','T') << std::endl;
                }
          }
        }
    }
    return 0;
}
