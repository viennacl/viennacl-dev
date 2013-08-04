
include_directories(${Boost_INCLUDE_DIRS})

# tests with CPU backend
foreach(PROG blas3_prod_float blas3_prod_double blas3_solve_float blas3_solve_double iterators
             global_variables
             matrix_vector
             matrix_row_float matrix_row_double matrix_col_float matrix_col_double
             scalar scheduler_matrix scheduler_vector sparse
             vector_float vector_double vector_int vector_multi_inner_prod
             spmdm)
   add_executable(${PROG}-test-cpu src/${PROG}.cpp)
   add_test(${PROG}-cpu ${PROG}-test-cpu)
endforeach(PROG)


# tests with OpenCL backend
if (ENABLE_OPENCL)
  foreach(PROG blas3_prod_float blas3_prod_double blas3_solve_float blas3_solve_double fft iterators
               generator_blas1 generator_blas2 generator_blas3 #generator_segmentation
               global_variables
               matrix_vector
               matrix_row_float matrix_row_double matrix_col_float matrix_col_double
               nmf qr_method
               scalar sparse structured-matrices svd
               vector_float vector_double vector_int vector_multi_inner_prod
               spmdm)
     add_executable(${PROG}-test-opencl src/${PROG}.cpp)
     target_link_libraries(${PROG}-test-opencl ${OPENCL_LIBRARIES})
     add_test(${PROG}-opencl ${PROG}-test-opencl)
     set_target_properties(${PROG}-test-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  add_executable(external_linkage-opencl
                 src/external_1.cpp
                 src/external_2.cpp
                 ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
  target_link_libraries(external_linkage-opencl ${OPENCL_LIBRARIES})
  set_target_properties(external_linkage-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
endif (ENABLE_OPENCL)

# tests with CUDA backend
if (ENABLE_CUDA)
  foreach(PROG blas3_prod_float blas3_prod_double blas3_solve_float blas3_solve_double iterators
               global_variables
               matrix_vector
               matrix_row_float matrix_row_double matrix_col_float matrix_col_double
               scalar sparse
               vector_float vector_double vector_int vector_multi_inner_prod
               spmdm)
     cuda_add_executable(${PROG}-test-cuda src/${PROG}.cu)
     add_test(${PROG}-cuda ${PROG}-test-cuda)
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  cuda_add_executable(external_linkage-cuda
                      src/external_1.cu
                      src/external_2.cu
                      ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
endif (ENABLE_CUDA)

# test shared library
include_directories(${PROJECT_SOURCE_DIR}/libviennacl/include/)

if(ENABLE_CUDA)
  if(ENABLE_OPENCL)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DVIENNACL_WITH_OPENCL") #set flags before setting executable!
    cuda_add_executable(libviennacl-test src/libviennacl.cu)
    target_link_libraries(libviennacl-test viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    cuda_add_executable(libviennacl-test src/libviennacl.cu)
    target_link_libraries(libviennacl-test viennacl)
  endif(ENABLE_OPENCL)
else(ENABLE_CUDA)
  add_executable(libviennacl-test src/libviennacl.cpp)
  if(ENABLE_OPENCL)
    set_target_properties(libviennacl-test PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(libviennacl-test viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    target_link_libraries(libviennacl-test viennacl)
  endif(ENABLE_OPENCL)
endif(ENABLE_CUDA)
add_test(libviennacl libviennacl-test)


