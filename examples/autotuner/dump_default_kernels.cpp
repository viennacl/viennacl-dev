/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
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

#include <fstream>

#include "viennacl/matrix.hpp"
#include "viennacl/generator/utils.hpp"
#include "viennacl/generator/generate.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"

void dump_string_to_file(std::string const & filename, std::string const & str){
  std::ofstream ofs(filename.c_str());
  ofs << str << std::endl;
}

template<typename ScalarType>
void dump_gemm_kernel(std::string const & device_name)
{
    std::string scalartype_name = viennacl::generator::utils::type_to_string<ScalarType>::value();
    viennacl::matrix<ScalarType> A;
    viennacl::matrix<ScalarType> B;
    viennacl::matrix<ScalarType> C;
    ScalarType alpha;
    ScalarType beta;

    viennacl::scheduler::statement saa(C, viennacl::op_assign(), alpha*viennacl::linalg::prod(A,B) + beta*C);
    viennacl::scheduler::statement sta(C, viennacl::op_assign(), alpha*viennacl::linalg::prod(trans(A),B) + beta*C);
    viennacl::scheduler::statement sat(C, viennacl::op_assign(), alpha*viennacl::linalg::prod(A,trans(B)) + beta*C);
    viennacl::scheduler::statement stt(C, viennacl::op_assign(), alpha*viennacl::linalg::prod(trans(A),trans(B)) + beta*C);

    //OpenCL
    dump_string_to_file("gemm_aa_" + scalartype_name + "_" + device_name + ".cl", viennacl::generator::get_opencl_program_string(saa));
//    dump_string_to_file("gemm_ta_" + scalartype_name + "_" + device_name + ".cl", viennacl::generator::get_opencl_program_string(sta));
//    dump_string_to_file("gemm_at_" + scalartype_name + "_" + device_name + ".cl", viennacl::generator::get_opencl_program_string(sat));
//    dump_string_to_file("gemm_tt_" + scalartype_name + "_" + device_name + ".cl", viennacl::generator::get_opencl_program_string(stt));

    //CUDA
    dump_string_to_file("gemm_aa_" + scalartype_name + "_" + device_name + ".cu", viennacl::generator::get_cuda_device_code(saa));
//    dump_string_to_file("gemm_ta_" + scalartype_name + "_" + device_name + ".cu", viennacl::generator::get_cuda_program_string(sta));
//    dump_string_to_file("gemm_at_" + scalartype_name + "_" + device_name + ".cu", viennacl::generator::get_cuda_program_string(sat));
//    dump_string_to_file("gemm_tt_" + scalartype_name + "_" + device_name + ".cu", viennacl::generator::get_cuda_program_string(stt));
}

int main(){
  typedef std::vector< viennacl::ocl::platform > platforms_type;
  unsigned int counter = 0;
  platforms_type platforms = viennacl::ocl::get_platforms();
  for (platforms_type::iterator platform_iter  = platforms.begin();
       platform_iter != platforms.end();
       ++platform_iter)
  {
    typedef std::vector<viennacl::ocl::device> devices_type;
    devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
    for(devices_type::iterator iter = devices.begin(); iter != devices.end(); iter++)
    {
      unsigned int current_device = counter++;
      viennacl::ocl::setup_context(current_device,*iter);
      viennacl::ocl::switch_context(current_device);
      viennacl::ocl::device const & device = viennacl::ocl::current_device();
      std::string device_name = device.name();
      std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::tolower);
      std::replace(device_name.begin(), device_name.end(),' ', '_');
      dump_gemm_kernel<float>(device_name);
    }
  }
}
