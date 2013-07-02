# Targets using CPU-based execution
foreach(bench blas3 scheduler)
   add_executable(${bench}bench-cpu ${bench}.cpp)
endforeach()

if (ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(bench sparse solver vector)
      add_executable(${bench}bench-cpu ${bench}.cpp)
    endforeach()
endif (ENABLE_UBLAS)


# Targets using OpenCL
if (ENABLE_OPENCL)

  foreach(bench generator_blas1 generator_blas2 generator_blas3 blas3 opencl)
    add_executable(${bench}bench-opencl ${bench}.cpp)
    target_link_libraries(${bench}bench-opencl ${OPENCL_LIBRARIES})
    set_target_properties(${bench}bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach()

  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver vector)
       add_executable(${bench}bench-opencl ${bench}.cpp)
       target_link_libraries(${bench}bench-opencl ${OPENCL_LIBRARIES})
       set_target_properties(${bench}bench-opencl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     endforeach()
  endif (ENABLE_UBLAS)

endif (ENABLE_OPENCL)

# Targets using CUDA
if (ENABLE_CUDA)

  foreach(bench blas3)
     cuda_add_executable(${bench}bench-cuda ${bench}.cu)
  endforeach()

  if (ENABLE_UBLAS)
     include_directories(${Boost_INCLUDE_DIRS})
     foreach(bench sparse solver vector)
       cuda_add_executable(${bench}bench-cuda ${bench}.cu)
     endforeach()
  endif (ENABLE_UBLAS)

endif (ENABLE_CUDA)


# IF(CMAKE_COMPILER_IS_GNUCXX)
   #ADD_DEFINITIONS(-Wall -pedantic -O0 -g)
#   ADD_DEFINITIONS(-Wall -pedantic -O3)
# ENDIF(CMAKE_COMPILER_IS_GNUCXX)
