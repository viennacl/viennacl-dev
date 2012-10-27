foreach(PROG blas3_prod blas3_solve fft iterators
        generator_inner_product generator_matrix_vector_product generator_vector generator_multiline
        global_variables
        matrix matrix_range matrix_slice nmf qr-method
        scalar sparse structured-matrices svd
        vector)
   add_executable(${PROG}-test src/${PROG}.cpp)
   target_link_libraries(${PROG}-test ${OPENCL_LIBRARIES})
   add_test(${PROG} ${PROG}-test)
endforeach(PROG)

include_directories(${PROJECT_SOURCE_DIR}/external)
add_executable(external_linkage 
                src/external_1.cpp
                src/external_2.cpp 
                ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
target_link_libraries(external_linkage ${OPENCL_LIBRARIES})
