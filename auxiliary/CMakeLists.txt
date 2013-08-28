include_directories(${Boost_INCLUDE_DIRS})


set(COMPRESSED_MATRIX_SRCS
   compressed_matrix/align1/block_trans_lu_backward.cl
   compressed_matrix/align1/block_trans_unit_lu_forward.cl
   compressed_matrix/align1/jacobi.cl
   compressed_matrix/align1/lu_backward.cl
   compressed_matrix/align1/lu_forward.cl
   compressed_matrix/align1/trans_lu_backward.cl
   compressed_matrix/align1/trans_lu_forward.cl
   compressed_matrix/align1/trans_unit_lu_backward.cl
   compressed_matrix/align1/trans_unit_lu_forward.cl
   compressed_matrix/align1/trans_unit_lu_forward_slow.cl
   compressed_matrix/align1/row_info_extractor.cl
   compressed_matrix/align1/unit_lu_backward.cl
   compressed_matrix/align1/unit_lu_forward.cl
   compressed_matrix/align1/vec_mul.cl
   compressed_matrix/align1/vec_mul_cpu.cl
   compressed_matrix/align1/d_mat_mul.cl
   compressed_matrix/align1/d_tr_mat_mul.cl
   compressed_matrix/align4/vec_mul.cl
   compressed_matrix/align8/vec_mul.cl)

set(COORDINATE_MATRIX_SRCS
   coordinate_matrix/align1/vec_mul.cl
   coordinate_matrix/align1/row_info_extractor.cl
   coordinate_matrix/align128/dummy)

set(HYB_MATRIX_SRCS
   hyb_matrix/align1/vec_mul.cl)

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

set(SPAI_SRCS
   spai/align1/assemble_blocks.cl
   spai/align1/block_bv_assembly.cl
   spai/align1/block_least_squares.cl
   spai/align1/block_q_mult.cl
   spai/align1/block_qr.cl
   spai/align1/block_qr_assembly.cl
   spai/align1/block_qr_assembly_1.cl
   spai/align1/block_r_assembly.cl
   )


set(RAND_SRCS
    rand/align1/_mwc64x.cl
    rand/align1/dump_gaussian.cl
    rand/align1/dump_uniform.cl
    )

set(CL_SRCS)
foreach(f IN LISTS COMPRESSED_MATRIX_SRCS COORDINATE_MATRIX_SRCS HYB_MATRIX_SRCS
      SVD_SRCS SPAI_SRCS
      RAND_SRCS)
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
      compressed_matrix
      coordinate_matrix
      hyb_matrix
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
