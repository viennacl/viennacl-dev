include_directories(${PROJECT_SOURCE_DIR}/external)

if (ENABLE_OPENCL)
  foreach(proc blas1 blas2 blas3)
    set(n ${proc}_tuning)
    add_executable(${n} ${proc}.cpp)
    target_link_libraries(${n} ${OPENCL_LIBRARIES})
  endforeach()
endif (ENABLE_OPENCL)
