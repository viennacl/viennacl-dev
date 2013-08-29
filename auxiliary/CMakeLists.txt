include_directories(${Boost_INCLUDE_DIRS})


set(RAND_SRCS
    rand/align1/_mwc64x.cl
    rand/align1/dump_gaussian.cl
    rand/align1/dump_uniform.cl
    )

set(CL_SRCS)
foreach(f IN LISTS RAND_SRCS)
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
