
include_directories(${PROJECT_SOURCE_DIR}/libviennacl/include/)

if(ENABLE_CUDA)
  if(ENABLE_OPENCL)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DVIENNACL_WITH_OPENCL") #set flags before setting executable!
  endif(ENABLE_OPENCL)

  cuda_add_library(viennacl SHARED src/blas1_host.cu src/blas1_cuda.cu src/blas1_opencl.cu)
  if(ENABLE_OPENCL)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_CUDA")
    target_link_libraries(viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_CUDA")
  endif(ENABLE_OPENCL)
else(ENABLE_CUDA)
  add_library(viennacl SHARED src/blas1_host.cpp src/blas1_opencl.cpp)
  if(ENABLE_OPENCL)
    set_target_properties(viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(viennacl ${OPENCL_LIBRARIES})
  endif(ENABLE_OPENCL)
endif(ENABLE_CUDA)
