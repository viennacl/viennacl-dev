
include_directories(${Boost_INCLUDE_DIRS})

# tests with CPU backend
foreach(PROG blas3_prod blas3_solve iterators
             global_variables
             matrix-vector matrix
             scalar scheduler_matrix scheduler_vector sparse
             vector)
   add_executable(${PROG}-test-cpu src/${PROG}.cpp)
   add_test(${PROG}-cpu ${PROG}-test-cpu)
endforeach(PROG)


# tests with OpenCL backend
if (ENABLE_OPENCL)
  foreach(PROG blas3_prod blas3_solve fft iterators
               generator_blas1 generator_blas2 generator_blas3 #generator_segmentation
               global_variables
               matrix-vector matrix nmf qr-method
               scalar sparse structured-matrices svd
               vector)
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
  foreach(PROG blas3_prod blas3_solve iterators
               global_variables
               matrix-vector matrix
               scalar sparse
               vector)
     cuda_add_executable(${PROG}-test-cuda src/${PROG}.cu)
     add_test(${PROG}-cuda ${PROG}-test-cuda)
  endforeach(PROG)

  include_directories(${PROJECT_SOURCE_DIR}/external)
  cuda_add_executable(external_linkage-cuda
                      src/external_1.cu
                      src/external_2.cu
                      ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
endif (ENABLE_CUDA)

