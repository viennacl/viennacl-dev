


#
# Part 1: Tutorials which work without OpenCL as well:
#
foreach(tut bandwidth-reduction blas1)
   add_executable(${tut} ${tut}.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(${tut} ${OPENCL_LIBRARIES})
     set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
   endif (ENABLE_OPENCL)
endforeach()

if(ENABLE_UBLAS)
   include_directories(${Boost_INCLUDE_DIRS})
   foreach(tut blas2 blas3 iterative-ublas lanczos least-squares matrix-range power-iter qr sparse vector-range)
      add_executable(${tut} ${tut}.cpp)
      target_link_libraries(${tut} ${Boost_LIBRARIES})
      if (ENABLE_OPENCL)
        target_link_libraries(${tut} ${OPENCL_LIBRARIES})
        set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
      endif (ENABLE_OPENCL)
   endforeach()
endif()

if(ENABLE_EIGEN)
   include_directories(${EIGEN_INCLUDE_DIR})
   add_executable(iterative-eigen     iterative-eigen.cpp)
   add_executable(eigen-with-viennacl eigen-with-viennacl.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(eigen-with-viennacl ${OPENCL_LIBRARIES})
     set_target_properties(eigen-with-viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
   endif (ENABLE_OPENCL)
endif()

if(ENABLE_MTL4)
   foreach(tut iterative-mtl4 mtl4-with-viennacl)
     add_executable(${tut} ${tut}.cpp)
     if (ENABLE_OPENCL)
       target_link_libraries(${tut} ${OPENCL_LIBRARIES})
       set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     endif (ENABLE_OPENCL)
   endforeach()
endif()



#
# Part 2: Tutorials which work only with OpenCL enabled:
#
if (ENABLE_OPENCL)
  foreach(tut custom-kernels custom-context fft viennacl-info)
    add_executable(${tut} ${tut}.cpp)
    target_link_libraries(${tut} ${OPENCL_LIBRARIES})
    set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach()
  
  if(ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(tut amg iterative spai structured-matrices)
        add_executable(${tut} ${tut}.cpp)
        if (ENABLE_OPENCL)
          target_link_libraries(${tut} ${OPENCL_LIBRARIES})
          set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
        endif (ENABLE_OPENCL)
    endforeach()
  endif()
endif (ENABLE_OPENCL)


#
# Part 3: Tutorials with CUDA support
# 

if (ENABLE_CUDA)
  foreach(tut blas1)
    cuda_add_executable(${tut}-cuda ${tut}.cu)
  endforeach()

  if(ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(tut blas2 blas3 iterative least-squares matrix-range power-iter qr sparse vector-range)
        cuda_add_executable(${tut}-cuda ${tut}.cu)
    endforeach()
  endif()

endif (ENABLE_CUDA)

