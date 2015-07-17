#
# Simple script which checks all header files in the ViennaCL source tree for being self-sufficient.
# Known to work with Python 2.7.3. Run from auxiliary/ folder.
# Requires g++ and nvcc to be in PATH as well as write permissions to this folder.
#
# License: MIT/X11, see LICENSE file in top level directory
#

import os
import subprocess

#
# Checks all files in 'files' to be self-sufficient by creating a simple source file with contents:
#   #include <file>
#   int main() {}
# and hands it over to the compiler. In the case of an error, the compiler output is printed and an exception thrown.
#
def run_include_check(files, compiler_options, compiler_name = "g++"):
  for filename in files:
    #write simple checker file and compile:
    print "Testing " + filename
    sourcefilename = "test-self-sufficient.cpp"
    if "nvcc" in compiler_name:
      sourcefilename = "test-self-sufficient.cu"
    file = open(sourcefilename, "w")
    file.write('#include "' + filename + '"' + "\n")
    file.write("int main() { return 0; }")
    file.close()
    try:
      subprocess.check_output(compiler_name + " " + sourcefilename + " " + compiler_options, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
      print "ERROR: ",e.output
      raise

#
# Iterates through
#
def run_check(root_path, exclude_dirs, exclude_files, compiler_options, compiler_name = "g++"):
  files_to_check=[]
  for root, dirs, files in os.walk(root_path):
    for filename in files:
      if not root in exclude_dirs:
        if not filename in exclude_files:
         files_to_check.append(os.path.join(root, filename))

  run_include_check(files_to_check, compiler_options, compiler_name)

###

rootPath = '../viennacl'

print " --- Vanilla tests (no OpenCL, no CUDA) --- "

exclude_files_vanilla=[
 "amg.hpp",
 "circulant_matrix.hpp",
 "circulant_matrix_operations.hpp",
 "cuda.hpp",
 "fft.hpp",
 "hankel_matrix.hpp",
 "hankel_matrix_operations.hpp",
 "vandermonde_matrix.hpp",
 "vandermonde_matrix_operations.hpp",
 "toeplitz_matrix.hpp",
 "toeplitz_matrix_operations.hpp",
 "mixed_precision_cg.hpp",
 "nmf.hpp",
 "opencl.hpp",
 "preset.hpp",
 "qr-method.hpp",
 "qr-method-common.hpp",
 "spai.hpp",
 "svd.hpp"]

exclude_dirs_vanilla=[
 "../viennacl/ocl",
 "../viennacl/linalg/cuda",
 "../viennacl/linalg/opencl",
 "../viennacl/linalg/opencl/kernels",
 "../viennacl/device_specific",
 "../viennacl/device_specific/builtin_database",
 "../viennacl/device_specific/builtin_database/devices",
 "../viennacl/device_specific/builtin_database/devices/accelerator",
 "../viennacl/device_specific/builtin_database/devices/cpu",
 "../viennacl/device_specific/builtin_database/devices/gpu",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla",
 "../viennacl/device_specific/templates",
 "../viennacl/linalg/detail/amg",
 "../viennacl/linalg/detail/spai",
 "../viennacl/rand"]
  
run_check(rootPath, exclude_dirs_vanilla, exclude_files_vanilla, "-I..")

###########################

print " --- OpenCL tests (no CUDA) --- "

exclude_files_opencl=["cuda.hpp",
 "matrix_operations.hpp",
 "vector_operations.hpp"]

exclude_dirs_opencl=[
 "../viennacl/linalg/cuda"
]
  
run_check(rootPath, exclude_dirs_opencl, exclude_files_opencl, "-I.. -DVIENNACL_WITH_OPENCL -lOpenCL")

###########################


print " --- CUDA tests (no OpenCL) --- "

exclude_files_cuda=[
 "amg.hpp",
 "circulant_matrix.hpp",
 "circulant_matrix_operations.hpp",
 "fft.hpp",
 "hankel_matrix.hpp",
 "hankel_matrix_operations.hpp",
 "vandermonde_matrix.hpp",
 "vandermonde_matrix_operations.hpp",
 "toeplitz_matrix.hpp",
 "toeplitz_matrix_operations.hpp",
 "mixed_precision_cg.hpp",
 "nmf.hpp",
 "opencl.hpp",
 "preset.hpp",
 "qr-method.hpp",
 "qr-method-common.hpp",
 "spai.hpp",
 "spgemm.hpp",
 "svd.hpp"]

exclude_dirs_cuda=[
 "../viennacl/ocl",
 "../viennacl/linalg/opencl",
 "../viennacl/linalg/opencl/kernels",
 "../viennacl/device_specific",
 "../viennacl/device_specific/builtin_database",
 "../viennacl/device_specific/builtin_database/devices",
 "../viennacl/device_specific/builtin_database/devices/accelerator",
 "../viennacl/device_specific/builtin_database/devices/cpu",
 "../viennacl/device_specific/builtin_database/devices/gpu",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/evergreen",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/northern_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/southern_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/amd/volcanic_islands",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/fermi",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/kepler",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/maxwell",
 "../viennacl/device_specific/builtin_database/devices/gpu/nvidia/tesla",
 "../viennacl/device_specific/templates",
 "../viennacl/linalg/detail/amg",
 "../viennacl/linalg/detail/spai",
 "../viennacl/rand"]
  
run_check(rootPath, exclude_dirs_cuda, exclude_files_cuda, " -I.. -DVIENNACL_WITH_CUDA ", "nvcc")

