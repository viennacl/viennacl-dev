include_directories(${PROJECT_SOURCE_DIR}/external)

if (ENABLE_OPENCL)
  foreach(proc dot gemv gemm)
    add_executable(${proc}_tuning ${proc}.cpp)
    target_link_libraries(${proc}_tuning ${OPENCL_LIBRARIES})
  endforeach()
endif (ENABLE_OPENCL)
