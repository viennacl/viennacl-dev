foreach(PROG blas3 blas3range fft iterators
        generator_inner_product generator_matrix_vector_product generator_vector generator_multiline
        matrix matrix_range matrix_slice nmf qr-method
        scalar sparse structured-matrices svd
        vector vector_range vector_slice)
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
