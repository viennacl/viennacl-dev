foreach(bench vector)
   add_executable(${bench}bench ${bench}.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(${bench}bench ${OPENCL_LIBRARIES})
     set_target_properties(${bench}bench PROPERTIES COMPILE_FLAGS "-DVIENNACL_HAVE_OPENCL")
   endif (ENABLE_OPENCL)
endforeach()

if (ENABLE_OPENCL)

  foreach(bench blas3 opencl)
    add_executable(${bench}bench ${bench}.cpp)
    target_link_libraries(${bench}bench ${OPENCL_LIBRARIES})
    set_target_properties(${bench}bench PROPERTIES COMPILE_FLAGS "-DVIENNACL_HAVE_OPENCL")
  endforeach()
  
  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver)
       add_executable(${bench}bench ${bench}.cpp)
       target_link_libraries(${bench}bench ${OPENCL_LIBRARIES})
       set_target_properties(${bench}bench PROPERTIES COMPILE_FLAGS "-DVIENNACL_HAVE_OPENCL")
     endforeach()
  endif (ENABLE_UBLAS)
  
endif (ENABLE_OPENCL)


# IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -pedantic -O0 -g)
#   ADD_DEFINITIONS(-Wall -pedantic -O3)
# ENDIF(CMAKE_COMPILER_IS_GNUCXX)
