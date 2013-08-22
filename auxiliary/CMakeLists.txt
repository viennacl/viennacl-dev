include_directories(${Boost_INCLUDE_DIRS})

add_executable(generate-blas3-solve-align1  generate-blas3-solve-align1.cpp)
add_executable(generate-blas3-prod-align1   generate-blas3-prod-align1.cpp)
add_executable(generate-blas3-prod16-align1 generate-blas3-prod16-align1.cpp)
add_executable(generate-blas3-prod-amd      generate-blas3-prod-amd.cpp)

function(generate_blas3_prod_align1 outvar)
   set(crstr_0 col)
   set(crstr_1 row)
   set(ATstr_0 A)
   set(ATstr_1 T)
   set(outfiles)

   foreach(ar 0 1) # A is column/row major
   foreach(br 0 1) # B is column/row major
   foreach(cr 0 1) # C is column/row major
   foreach(at 0 1) # A is (not) transposed
   foreach(bt 0 1) # B is (not) transposed
      set(d "${CMAKE_CURRENT_BINARY_DIR}")
      set(d "${d}/matrix_prod_${crstr_${ar}}_${crstr_${br}}_${crstr_${cr}}")
      set(d "${d}/align1")
      file(MAKE_DIRECTORY "${d}")

      # standard kernels:
      set(o "${d}/prod_${ATstr_${at}}${ATstr_${bt}}.cl")
      file(RELATIVE_PATH ro "${CMAKE_CURRENT_BINARY_DIR}" "${o}")
      add_custom_command(OUTPUT "${o}"
         COMMAND generate-blas3-prod-align1
            ${ar} ${br} ${cr} ${at} ${bt} 0 > "${o}"
         COMMENT "Generating ${ro}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o}")

      # fast kernels:
      set(o16 "${d}/prod16_${ATstr_${at}}${ATstr_${bt}}.cl")
      file(RELATIVE_PATH ro16 "${CMAKE_CURRENT_BINARY_DIR}" "${o16}")
      add_custom_command(OUTPUT "${o16}"
         COMMAND generate-blas3-prod16-align1
            ${ar} ${br} ${cr} ${at} ${bt} > "${o16}"
         COMMENT "Generating ${ro16}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o16}")

      # AMD Tahiti optimized kernels:
      set(o "${d}/prod_${ATstr_${at}}${ATstr_${bt}}_amd.cl")
      file(RELATIVE_PATH ro "${CMAKE_CURRENT_BINARY_DIR}" "${o}")
      add_custom_command(OUTPUT "${o}"
         COMMAND generate-blas3-prod-amd
            ${ar} ${br} ${cr} ${at} ${bt} > "${o}"
         COMMENT "Generating ${ro}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o}")

   endforeach()
   endforeach()
   endforeach()
   endforeach()
   endforeach()
   set(${outvar} "${outfiles}" PARENT_SCOPE)
endfunction()

function(generate_blas3_prod_amd outvar)
   set(crstr_0 col)
   set(crstr_1 row)
   set(ATstr_0 A)
   set(ATstr_1 T)
   set(outfiles)

   foreach(ar 0 1) # A is column/row major
   foreach(br 0 1) # B is column/row major
   foreach(cr 0 1) # C is column/row major
   foreach(at 0 1) # A is (not) transposed
   foreach(bt 0 1) # B is (not) transposed
      set(d "${CMAKE_CURRENT_BINARY_DIR}")
      set(d "${d}/matrix_prod_${crstr_${ar}}_${crstr_${br}}_${crstr_${cr}}")
      set(d "${d}/align1")
      file(MAKE_DIRECTORY "${d}")

      # standard kernels:
      set(o "${d}/prod_${ATstr_${at}}${ATstr_${bt}}_amd.cl")
      file(RELATIVE_PATH ro "${CMAKE_CURRENT_BINARY_DIR}" "${o}")
      add_custom_command(OUTPUT "${o}"
         COMMAND generate-blas3-prod-align1
            ${ar} ${br} ${cr} ${at} ${bt} 0 > "${o}"
         COMMENT "Generating ${ro}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o}")

      # fast kernels:
      set(o16 "${d}/prod16_${ATstr_${at}}${ATstr_${bt}}.cl")
      file(RELATIVE_PATH ro16 "${CMAKE_CURRENT_BINARY_DIR}" "${o16}")
      add_custom_command(OUTPUT "${o16}"
         COMMAND generate-blas3-prod16-align1
            ${ar} ${br} ${cr} ${at} ${bt} > "${o16}"
         COMMENT "Generating ${ro16}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o16}")

   endforeach()
   endforeach()
   endforeach()
   endforeach()
   endforeach()
   set(${outvar} "${outfiles}" PARENT_SCOPE)
endfunction()

function(generate_blas3_solve_align1 outvar)
   set(crstr_0 col)
   set(crstr_1 row)
   set(tstr_0)
   set(tstr_1 trans_)
   set(ulstr_0 lower)
   set(ulstr_1 upper)
   set(unitstr_0)
   set(unitstr_1 unit_)
   set(outfiles)

   foreach(ar 0 1) # A is column/row major
   foreach(br 0 1) # A is column/row major
   foreach(at 0 1) # A is transposed
   foreach(bt 0 1) # B is transposed
   foreach(ul 0 1) # upper/lower
   foreach(un 0 1) # unit
      set(d "${CMAKE_CURRENT_BINARY_DIR}")
      set(d "${d}/matrix_solve_${crstr_${ar}}_${crstr_${br}}")
      set(d "${d}/align1")
      file(MAKE_DIRECTORY "${d}")
      set(o "${d}/${tstr_${at}}${unitstr_${un}}${ulstr_${ul}}_${tstr_${bt}}solve.cl")
      file(RELATIVE_PATH ro "${CMAKE_CURRENT_BINARY_DIR}" "${o}")
      add_custom_command(OUTPUT "${o}"
         COMMAND generate-blas3-solve-align1
            ${ar} ${br} ${at} ${bt} ${ul} ${un} > "${o}"
         COMMENT "Generating ${ro}"
         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
         VERBATIM)
      list(APPEND outfiles "${o}")
   endforeach()
   endforeach()
   endforeach()
   endforeach()
   endforeach()
   endforeach()
   set(${outvar} "${outfiles}" PARENT_SCOPE)
endfunction()

# Matrix-Matrix products
generate_blas3_prod_align1(MATRIX_PROD_SRCS)

# Matrix-Matrix triangular solver
generate_blas3_solve_align1(MATRIX_SOLVE_SRCS)

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

set(ELL_MATRIX_SRCS
   ell_matrix/align1/vec_mul.cl
   ell_matrix/align1/d_mat_mul.cl
   ell_matrix/align1/d_tr_mat_mul.cl)

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

set(NMF_SRCS
   nmf/align1/el_wise_mul_div.cl
   nmf/align1/sub_wise.cl
   )

set(ILU_SRCS
   ilu/align1/level_scheduling_substitute.cl
   )

set(RAND_SRCS
    rand/align1/_mwc64x.cl
    rand/align1/dump_gaussian.cl
    rand/align1/dump_uniform.cl
    )

set(CL_SRCS)
foreach(f IN LISTS COMPRESSED_MATRIX_SRCS COORDINATE_MATRIX_SRCS ELL_MATRIX_SRCS HYB_MATRIX_SRCS
      SVD_SRCS SPAI_SRCS NMF_SRCS ILU_SRCS
      RAND_SRCS)
   get_filename_component(d "${CMAKE_CURRENT_BINARY_DIR}/${f}" PATH)
   file(MAKE_DIRECTORY "${d}")
   configure_file(${f} "${CMAKE_CURRENT_BINARY_DIR}/${f}" COPYONLY)
   list(APPEND CL_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${f}")
endforeach()
list(APPEND CL_SRCS ${MATRIX_PROD_SRCS} ${MATRIX_SOLVE_SRCS})

configure_file(converter.cpp ${CMAKE_CURRENT_BINARY_DIR}/converter.cpp @ONLY)
add_executable(converter ${CMAKE_CURRENT_BINARY_DIR}/converter.cpp)
target_link_libraries(converter ${Boost_LIBRARIES})

set(KERNEL_HDRS)
set(KERNEL_SRCS)
foreach(d
      compressed_matrix
      coordinate_matrix
      ell_matrix
      hyb_matrix
      matrix_prod_col_col_col
      matrix_prod_col_col_row
      matrix_prod_col_row_col
      matrix_prod_col_row_row
      matrix_prod_row_col_col
      matrix_prod_row_col_row
      matrix_prod_row_row_col
      matrix_prod_row_row_row
      matrix_solve_col_col
      matrix_solve_col_row
      matrix_solve_row_col
      matrix_solve_row_row
      svd
      spai
      nmf
      ilu
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
