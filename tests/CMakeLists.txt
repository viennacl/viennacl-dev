
include_directories(${Boost_INCLUDE_DIRS})

# tests with CPU backend
foreach(PROG matrix_product_float matrix_product_double blas3_solve fft_1d fft_2d iterators
             global_variables
             nmf
             matrix_convert
             matrix_vector matrix_vector_int
             matrix_row_float matrix_row_double matrix_row_int
             matrix_col_float matrix_col_double matrix_col_int
             scalar scheduler_matrix scheduler_matrix_matrix self_assign qr_method qr_method_func scan scheduler_matrix_vector scheduler_sparse scheduler_vector sparse sparse_prod
             tql vector_convert vector_float_double vector_int vector_uint vector_multi_inner_prod
             spmdm)
   add_executable(${PROG}-test-cpu src/${PROG}.cpp)
   target_link_libraries(${PROG}-test-cpu ${Boost_LIBRARIES})
   add_test(${PROG}-cpu ${PROG}-test-cpu)
endforeach(PROG)


# tests with OpenCL backend
if (ENABLE_OPENCL)
  foreach(PROG bisect matrix_product_float matrix_product_double blas3_solve fft_1d fft_2d iterators
               global_variables
               matrix_convert
               matrix_vector matrix_vector_int
               matrix_row_float matrix_row_double matrix_row_int
               matrix_col_float matrix_col_double matrix_col_int
               nmf qr_method qr_method_func scan
               scalar self_assign sparse sparse_prod structured-matrices svd tql
               vector_convert vector_float_double vector_int vector_uint vector_multi_inner_prod
               spmdm)
     add_executable(${PROG}-test-opencl src/${PROG}.cpp)
     target_link_libraries(${PROG}-test-opencl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
     add_test(${PROG}-opencl ${PROG}-test-opencl)
     set_target_properties(${PROG}-test-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  add_executable(external_linkage-opencl
                 src/external_1.cpp
                 src/external_2.cpp)
  target_link_libraries(external_linkage-opencl ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  set_target_properties(external_linkage-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
endif (ENABLE_OPENCL)

# tests with CUDA backend
if (ENABLE_CUDA)
  foreach(PROG bisect matrix_product_float matrix_product_double blas3_solve fft_1d fft_2d iterators
               global_variables
               matrix_convert
               matrix_vector matrix_vector_int
               matrix_row_float matrix_row_double matrix_row_int
               matrix_col_float matrix_col_double matrix_col_int nmf
               scalar self_assign sparse qr_method qr_method_func scan sparse_prod tql
               vector_convert vector_float_double vector_int vector_uint vector_multi_inner_prod
               spmdm)
     cuda_add_executable(${PROG}-test-cuda src/${PROG}.cu)
     target_link_libraries(${PROG}-test-cuda ${Boost_LIBRARIES})
     add_test(${PROG}-cuda ${PROG}-test-cuda)
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  cuda_add_executable(external_linkage-cuda
                      src/external_1.cu
                      src/external_2.cu)
  target_link_libraries(external_linkage-cuda ${Boost_LIBRARIES})
endif (ENABLE_CUDA)

# test shared library
include_directories(${PROJECT_SOURCE_DIR}/libviennacl/include/)

if (ENABLE_CUDA)
  if (ENABLE_OPENCL)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DVIENNACL_WITH_OPENCL") #set flags before setting executable!
    cuda_add_executable(libviennacl_blas1-test src/libviennacl_blas1.cu)
    target_link_libraries(libviennacl_blas1-test viennacl ${OPENCL_LIBRARIES})

    cuda_add_executable(libviennacl_blas2-test src/libviennacl_blas2.cu)
    target_link_libraries(libviennacl_blas2-test viennacl ${OPENCL_LIBRARIES})

    cuda_add_executable(libviennacl_blas3-test src/libviennacl_blas3.cu)
    target_link_libraries(libviennacl_blas3-test viennacl ${OPENCL_LIBRARIES})

  else(ENABLE_OPENCL)
    cuda_add_executable(libviennacl_blas1-test src/libviennacl_blas1.cu)
    target_link_libraries(libviennacl_blas1-test viennacl)

    cuda_add_executable(libviennacl_blas2-test src/libviennacl_blas2.cu)
    target_link_libraries(libviennacl_blas2-test viennacl)

    cuda_add_executable(libviennacl_blas3-test src/libviennacl_blas3.cu)
    target_link_libraries(libviennacl_blas3-test viennacl)
  endif (ENABLE_OPENCL)
else(ENABLE_CUDA)
  add_executable(libviennacl_blas1-test src/libviennacl_blas1.cpp)
  add_executable(libviennacl_blas2-test src/libviennacl_blas2.cpp)
  add_executable(libviennacl_blas3-test src/libviennacl_blas3.cpp)
  if (ENABLE_OPENCL)
    set_target_properties(libviennacl_blas1-test PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(libviennacl_blas1-test viennacl ${OPENCL_LIBRARIES})

    set_target_properties(libviennacl_blas2-test PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(libviennacl_blas2-test viennacl ${OPENCL_LIBRARIES})

    set_target_properties(libviennacl_blas3-test PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(libviennacl_blas3-test viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    target_link_libraries(libviennacl_blas1-test viennacl)
    target_link_libraries(libviennacl_blas2-test viennacl)
    target_link_libraries(libviennacl_blas3-test viennacl)
  endif (ENABLE_OPENCL)
endif (ENABLE_CUDA)
add_test(libviennacl-blas1 libviennacl_blas1-test)
add_test(libviennacl-blas2 libviennacl_blas2-test)
add_test(libviennacl-blas3 libviennacl_blas3-test)


