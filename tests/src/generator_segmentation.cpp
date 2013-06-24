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
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/generator/custom_operation.hpp"

#define CHECK_RESULT(ncreated,nexpected,op) \
    if (ncreated != nexpected) {\
        std::cout << "# Error at operation: " #op << std::endl;\
        std::cout << "  Number of Kernels Created : " << ncreated << std::endl;\
        std::cout << "  Number of Kernels Expected : " << nexpected << std::endl;\
        retval = EXIT_FAILURE;\
    }\

template< typename NumericT, class Layout>
int test () {
    int retval = EXIT_SUCCESS;

    typedef viennacl::generator::vector<NumericT> vec;
    typedef viennacl::generator::scalar<NumericT> scal;
    typedef viennacl::generator::matrix<viennacl::matrix<NumericT, Layout> > mat;


    unsigned int size = 512;
    viennacl::vector<NumericT> w (size);
    viennacl::vector<NumericT> x (size);
    viennacl::vector<NumericT> y (size);
    viennacl::vector<NumericT> z (size);
    viennacl::scalar<NumericT> gs(0);
    viennacl::matrix<NumericT,Layout> A (size, size);
    viennacl::matrix<NumericT,Layout> B (size, size);
    viennacl::matrix<NumericT,Layout> C (size, size);

    {
      std::string name = "Single Inner Product...";
      std::cout << name << std::endl;
      unsigned int nkernels;
      viennacl::generator::custom_operation op;
      op.add(scal(gs) = viennacl::generator::inner_prod(vec(x),vec(y)));
      op.program(&nkernels);
      CHECK_RESULT(nkernels,2,name)
    }

    {
      std::string name = "Double Vector SAXPY...";
      std::cout << name << std::endl;
      unsigned int nkernels;
      viennacl::generator::custom_operation op;
      op.add(vec(x) = vec(y) + vec(z));
      op.add(vec(y) = vec(z) + vec(x));
      op.program(&nkernels);
      CHECK_RESULT(nkernels,1,name)
    }

    {
      std::string name = "Double Matrix SAXPY...";
      std::cout << name << std::endl;
      unsigned int nkernels;
      viennacl::generator::custom_operation op;
      op.add(mat(C) = mat(A) + mat(B));
      op.add(mat(A) = mat(C) + mat(B));
      op.program(&nkernels);
      CHECK_RESULT(nkernels,1,name)
    }

    {
      std::string name = "Double Vector SAXPY + Double Matrix SAXPY...";
      std::cout << name << std::endl;
      unsigned int nkernels;
      viennacl::generator::custom_operation op;
      op.add(vec(x) = vec(y) + vec(z));
      op.add(vec(y) = vec(z) + vec(x));
      op.add(mat(C) = mat(A) + mat(B));
      op.add(mat(A) = mat(C) + mat(B));
      op.program(&nkernels);
      CHECK_RESULT(nkernels,2,name)
    }

    {
      std::string name = "Double GEMV...";
      std::cout << name << std::endl;
      unsigned int nkernels;
      viennacl::generator::custom_operation op;
      op.add(vec(x) = viennacl::generator::prod(mat(A),vec(z)));
      op.add(vec(w) = viennacl::generator::prod(mat(B),vec(y)));
      op.program(&nkernels);
      CHECK_RESULT(nkernels,2,name)
    }

    return retval;
}

int main(int argc, char* argv[]){
    std::vector<std::string> args(argv,argv+argc);
    unsigned int requested_device;
    if(argc!=2){
        requested_device=0;
    }
    else{
        requested_device = atoi(args[1].c_str());
    }
    int retval = EXIT_SUCCESS;

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

                std::cout << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;
                std::cout << "## Test :: Operations Segmentation" << std::endl;
                std::cout << "----------------------------------------------" << std::endl;


               std::cout << "# Testing setup:" << std::endl;
               std::cout << "  numeric: float" << std::endl;
               retval = test<float, viennacl::row_major>();

               if ( retval == EXIT_SUCCESS )
                   std::cout << "# Test passed" << std::endl;
               else
                   return retval;



            }
        }
    }
    return EXIT_SUCCESS;
}
