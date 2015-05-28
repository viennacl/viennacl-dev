


#
# Part 1: Tutorials which work without OpenCL as well:
#
foreach(tut bandwidth-reduction blas1 scheduler fft wrap-host-buffer nmf)
   add_executable(${tut} ${tut}.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(${tut} ${OPENCL_LIBRARIES})
     set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
   endif (ENABLE_OPENCL)
endforeach()

if (ENABLE_UBLAS)
   include_directories(${Boost_INCLUDE_DIRS})
   foreach(tut blas2 blas3 iterative-ublas lanczos least-squares matrix-range power-iter qr sparse qr_method tql2 vector-range)
      add_executable(${tut} ${tut}.cpp)
      target_link_libraries(${tut} ${Boost_LIBRARIES})
      if (ENABLE_OPENCL)
        target_link_libraries(${tut} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
        set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
      endif (ENABLE_OPENCL)
   endforeach()
endif ()

if (ENABLE_ARMADILLO)
   include_directories(${ARMADILLO_INCLUDE_DIR})
   add_executable(iterative-armadillo     iterative-armadillo.cpp)
   add_executable(armadillo-with-viennacl armadillo-with-viennacl.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(armadillo-with-viennacl ${OPENCL_LIBRARIES})
     set_target_properties(armadillo-with-viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
   endif (ENABLE_OPENCL)
endif ()

if (ENABLE_EIGEN)
   include_directories(${EIGEN_INCLUDE_DIR})
   add_executable(iterative-eigen     iterative-eigen.cpp)
   add_executable(eigen-with-viennacl eigen-with-viennacl.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(eigen-with-viennacl ${OPENCL_LIBRARIES})
     set_target_properties(eigen-with-viennacl PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
   endif (ENABLE_OPENCL)
endif ()

if (ENABLE_MTL4)
   foreach(tut iterative-mtl4 mtl4-with-viennacl)
     add_executable(${tut} ${tut}.cpp)
     if (ENABLE_OPENCL)
       target_link_libraries(${tut} ${OPENCL_LIBRARIES})
       set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
     endif (ENABLE_OPENCL)
   endforeach()
endif ()



#
# Part 2: Tutorials which work only with OpenCL enabled:
#
if (ENABLE_OPENCL)
  foreach(tut bisect custom-kernels custom-context viennacl-info)
    add_executable(${tut} ${tut}.cpp)
    target_link_libraries(${tut} ${OPENCL_LIBRARIES})
    set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
  endforeach()

  if (ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(tut amg iterative multithreaded multithreaded_cg spai structured-matrices)
        add_executable(${tut} ${tut}.cpp)
        target_link_libraries(${tut} ${Boost_LIBRARIES})
        if (ENABLE_OPENCL)
          target_link_libraries(${tut} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
          set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
        endif (ENABLE_OPENCL)
    endforeach()
  endif ()
endif (ENABLE_OPENCL)


#
# Part 3: Tutorials with CUDA support
#

if (ENABLE_CUDA)
  foreach(tut blas1 fft nmf wrap-cuda-buffer)
    cuda_add_executable(${tut}-cuda ${tut}.cu)
  endforeach()
  cuda_add_executable(custom-cuda  custom-cuda.cu)  # would otherwise result in custom-cuda-cuda

  if (ENABLE_UBLAS)
    include_directories(${Boost_INCLUDE_DIRS})
    foreach(tut bisect blas2 blas3 iterative least-squares matrix-range power-iter qr qr_method sparse vector-range tql2 )
        cuda_add_executable(${tut}-cuda ${tut}.cu)
        target_link_libraries(${tut}-cuda ${Boost_LIBRARIES})
    endforeach()
  endif ()

endif (ENABLE_CUDA)

#
# Part 4: Tutorials on using the shared C-library
#

include_directories(${PROJECT_SOURCE_DIR}/libviennacl/include/)

if (ENABLE_CUDA)
  if (ENABLE_OPENCL)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-DVIENNACL_WITH_OPENCL") #set flags before setting executable!
    cuda_add_executable(libviennacl-tutorial libviennacl.cu)
    #set_target_properties(libviennacl-tutorial PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -DVIENNACL_WITH_CUDA")
    target_link_libraries(libviennacl-tutorial viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    cuda_add_executable(libviennacl-tutorial libviennacl.cu)
    target_link_libraries(libviennacl-tutorial viennacl)
  endif (ENABLE_OPENCL)
else(ENABLE_CUDA)
  add_executable(libviennacl-tutorial libviennacl.cpp)
  if (ENABLE_OPENCL)
    set_target_properties(libviennacl-tutorial PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL")
    target_link_libraries(libviennacl-tutorial viennacl ${OPENCL_LIBRARIES})
  else(ENABLE_OPENCL)
    target_link_libraries(libviennacl-tutorial viennacl)
  endif (ENABLE_OPENCL)
endif (ENABLE_CUDA)
