set(EXAMPLES_TESTDATA
   testdata/mat65k.mtx
   testdata/README
   testdata/result65025.txt
   testdata/rhs65025.txt)
foreach(f IN LISTS EXAMPLES_TESTDATA)
   configure_file(${f} "${CMAKE_CURRENT_BINARY_DIR}/${f}" COPYONLY)
endforeach()

add_subdirectory(tutorial)
add_subdirectory(autotuner)
add_subdirectory(benchmarks)
add_subdirectory(parameters)
