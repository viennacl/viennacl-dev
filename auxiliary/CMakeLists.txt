include_directories(${Boost_INCLUDE_DIRS})

set(SVD_SRCS
   svd/align1/bidiag_pack.cl
   svd/align1/col_reduce_lcl_array.cl
   svd/align1/copy_col.cl
   svd/align1/copy_row.cl
   svd/align1/final_iter_update.cl
   svd/align1/givens_next.cl
   svd/align1/givens_prev.cl
   svd/align1/house_update_A_left.cl
   svd/align1/house_update_A_right.cl
   svd/align1/house_update_QL.cl
   svd/align1/house_update_QR.cl
   svd/align1/inverse_signs.cl
   svd/align1/transpose_inplace.cl
   svd/align1/update_qr_column.cl
   )


set(RAND_SRCS
    rand/align1/_mwc64x.cl
    rand/align1/dump_gaussian.cl
    rand/align1/dump_uniform.cl
    )

set(CL_SRCS)
foreach(f IN LISTS SVD_SRCS SPAI_SRCS RAND_SRCS)
   get_filename_component(d "${CMAKE_CURRENT_BINARY_DIR}/${f}" PATH)
   file(MAKE_DIRECTORY "${d}")
   configure_file(${f} "${CMAKE_CURRENT_BINARY_DIR}/${f}" COPYONLY)
   list(APPEND CL_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${f}")
endforeach()
list(APPEND CL_SRCS ${MATRIX_PROD_SRCS})

configure_file(converter.cpp ${CMAKE_CURRENT_BINARY_DIR}/converter.cpp @ONLY)
add_executable(converter ${CMAKE_CURRENT_BINARY_DIR}/converter.cpp)
target_link_libraries(converter ${Boost_LIBRARIES})

set(KERNEL_HDRS)
set(KERNEL_SRCS)
foreach(d
      svd
      spai
      rand
      )
   set(f "${PROJECT_BINARY_DIR}/viennacl/linalg/kernels/${d}")
   list(APPEND KERNEL_HDRS "${f}_kernels.h")
   list(APPEND KERNEL_SRCS "${f}_source.h")
endforeach()
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/viennacl/linalg/kernels")

add_custom_command(OUTPUT ${KERNEL_HDRS} ${KERNEL_SRCS}
   COMMAND converter
   DEPENDS ${CL_SRCS}
   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
   COMMENT "Generating kernel headers and sources"
   VERBATIM)

add_custom_target(kernels ALL
   DEPENDS ${KERNEL_HDRS} ${KERNEL_SRCS})
