include_directories(${PROJECT_SOURCE_DIR}/external)

if (ENABLE_OPENCL)
  include_directories(${PROJECT_SOURCE_DIR}/external/)

  foreach(proc gemm_autotuning)
    add_executable(${proc} ${proc}.cpp)
    target_link_libraries(${proc} ${OPENCL_LIBRARIES})
    set_target_properties(${proc} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach()

endif (ENABLE_OPENCL)
