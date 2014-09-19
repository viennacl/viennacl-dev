set(EXAMPLES_TESTDATA
   testdata/mat65k.mtx
   testdata/README
   testdata/result65025.txt
   testdata/rhs65025.txt
   testdata/eigen/nsm1.example
   testdata/eigen/nsm2.example
   testdata/eigen/nsm3.example
   testdata/eigen/symm5.example
   testdata/svd/qr.example
   testdata/svd/wiki.example
   testdata/svd/wiki.qr.example
   testdata/svd/pysvd.example
   testdata/svd/random.example)
foreach(f ${EXAMPLES_TESTDATA})
   configure_file(${f} "${CMAKE_CURRENT_BINARY_DIR}/${f}" COPYONLY)
endforeach()

add_subdirectory(tutorial)
add_subdirectory(benchmarks)
