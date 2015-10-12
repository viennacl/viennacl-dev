# variables for tests
CC           = clang-omp++
CFLAGS       = -Wall -pedantic -std=c++11 -O3 -DVIENNACL_WITH_OPENMP -fopenmp
AVX_FLAGS    = -mavx -DVIENNACL_WITH_AVX
SSE_FLAGS    = -DVIENNACL_WITH_SSE -msse3
LIBS         = -I/Users/max/viennacl-dev -I/Users/max/viennacl-dev/libviennacl/include
DEP_PRODTEST = src/prod_test.cpp ../viennacl/linalg/host_based/matrix_operations.hpp ../viennacl/linalg/host_based/packing.hpp ../viennacl/linalg/host_based/gemm_avx_micro_kernel.hpp ../viennacl/linalg/host_based/get_block_sizes.hpp ../viennacl/linalg/host_based/aligned_buffer.hpp

# variables for benchmarks
OPENBLAS_LIBS = -I/Users/max/openblas/ -L/Users/max/openblas/ -lopenblas -lpthread
EIGEN_LIBS    = -I/Users/max/Eigen


all: prod_test avx_prod_test avx_prod_test2

# test targets
astest.s: src/astest.cpp
	$(CC) src/astest.cpp -S build/astest.cpp -g -Wall -pedantic -std=c++11

astest: src/astest.cpp src/astest.s
	$(CC) src/astest.cpp -o build/astest -g -Wall -pedantic -std=c++11

test: src/test.cpp
	$(CC) src/test.cpp -o build/test $(CFLAGS) 

avx_prod_test: $(DEP_PRODTEST)
	$(CC) src/prod_test.cpp -o build/avx_prod_test $(CFLAGS) $(LIBS) $(AVX_FLAGS)

avx_prod_test2: $(DEP_PRODTEST)
	$(CC) src/prod_test.cpp -o build/avx_prod_test2 $(CFLAGS) $(LIBS) $(AVX_FLAGS) -g

sse_prod_test: $(DEP_PRODTEST)
	$(CC) src/prod_test.cpp -o build/sse_prod_test $(CFLAGS) $(LIBS) -DVIENNACL_WITH_SSE -msse3

prod_test: $(DEP_PRODTEST)
	$(CC) src/prod_test.cpp -o build/prod_test $(CFLAGS) $(LIBS)

# benchmark targets
viennacl: viennacl_standard viennacl_avx

viennacl_standard: src/benchmark_viennacl.cpp
	$(CC) src/benchmark_viennacl.cpp -o build/bench_viennacl $(CFLAGS) $(LIBS)

viennacl_avx: src/benchmark_viennacl.cpp
	$(CC) src/benchmark_viennacl.cpp -o build/bench_viennacl_avx $(CFLAGS) $(LIBS) $(AVX_FLAGS)

viennacl_sse: src/benchmark_viennacl.cpp
	$(CC) src/benchmark_viennacl.cpp -o build/bench_viennacl_sse $(CFLAGS) $(LIBS) $(SSE_FLAGS)

eigen: src/benchmark_eigen.cpp
	$(CC) src/benchmark_eigen.cpp -o build/bench_eigen $(CFLAGS) $(LIBS) $(EIGEN_LIBS)

openblas: src/benchmark_openblas.cpp
	$(CC) src/benchmark_openblas.cpp -o build/bench_openblas $(CFLAGS) $(LIBS) $(OPENBLAS_LIBS)

clean:
	cd build && rm test prod_test avx_prod_test astest bench_viennacl bench_viennacl_avx bench_viennacl_avx2 bench_eigen bench_openblas astest.s
