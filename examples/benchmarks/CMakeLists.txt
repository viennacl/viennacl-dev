foreach(bench vector blas3)
   add_executable(${bench}bench-cpu ${bench}.cpp)
endforeach()

if (ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(bench sparse)
      add_executable(${bench}bench-cpu ${bench}.cpp)
    endforeach()
endif (ENABLE_UBLAS)
  

if (ENABLE_OPENCL)

  foreach(bench blas3 opencl)
    add_executable(${bench}bench-opencl ${bench}.cpp)
    target_link_libraries(${bench}bench-opencl ${OPENCL_LIBRARIES})
    set_target_properties(${bench}bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach()
  
  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver)
       add_executable(${bench}bench-opencl ${bench}.cpp)
       target_link_libraries(${bench}bench-opencl ${OPENCL_LIBRARIES})
       set_target_properties(${bench}bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     endforeach()
  endif (ENABLE_UBLAS)
  
endif (ENABLE_OPENCL)


# IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -pedantic -O0 -g)
#   ADD_DEFINITIONS(-Wall -pedantic -O3)
# ENDIF(CMAKE_COMPILER_IS_GNUCXX)
