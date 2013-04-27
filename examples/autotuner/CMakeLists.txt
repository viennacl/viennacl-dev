include_directories(${PROJECT_SOURCE_DIR}/external)

if (ENABLE_OPENCL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  foreach(par inner_product saxpy)
    set(n ${par}_tuning)
    if(par STREQUAL parameter_reader)
      set(n ${par})
    endif()
    add_executable(${n} ${par}.cpp
      ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
    target_link_libraries(${n} ${OPENCL_LIBRARIES})
  endforeach()

  if(ENABLE_VIENNAPROFILER)
    include_directories(${VIENNAPROFILER_INCLUDE_DIRS})
    foreach(par vector_tuning)
      add_executable(${par}params_vprof ${par}.cpp
          ${PROJECT_SOURCE_DIR}/external/pugixml/src/pugixml.cpp)
      target_link_libraries(${par}params_vprof
          ${OPENCL_LIBRARIES} ${VIENNAPROFILER_LIBRARIES})
    endforeach()

    set_target_properties(vectorparams_vprof matrixparams_vprof
      sparseparams_vprof
      PROPERTIES COMPILE_DEFINITIONS ENABLE_VIENNAPROFILER)
  endif()
endif (ENABLE_OPENCL)
