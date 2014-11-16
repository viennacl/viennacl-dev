[![Build Status](https://travis-ci.org/viennacl/viennacl-dev.svg?branch=master)](https://travis-ci.org/viennacl/viennacl-dev)
Developer Repository for ViennaCL
==========================================

*Looking for ViennaCL releases? Visit [http://viennacl.sourceforge.net/](http://viennacl.sourceforge.net/)*

This is the developer repository of ViennaCL including the latest features and changes. Feel free to clone the repository, send us your pull requests, or update the Wiki here at github. All contributions are welcome. You might also want to subscribe to our [developer mailing list](http://lists.sourceforge.net/mailman/listinfo/viennacl-devel). There are no 'stupid questions', so don't hesitate to get in contact with us.

To build the developer version of ViennaCL, simply clone the repository and issue the following commands (the following steps are for Unix-based systems):
<pre>
$> cd viennacl-dev
$> mkdir build && cd build
$> cmake ..
$> make
</pre>

(Feel free to use parallel builds through `make -j4`, but keep in mind that each build might take up to one GB of RAM)

Follow similar steps on Windows:

* Launch the CMake-GUI and point the source-directory to `viennacl-dev` and the build-directory to `viennacl-dev/build`.
* Confirm that CMake should create the build-folder for you.
* Click on `Configure` and select your compilation environment.
* Provide any missing paths to Boost and/or OpenCL, or deselect `ENABLE_UBLAS` and/or `ENABLE_OPENCL`.
* Click on `Configure` again and then on `Generate`.
* You will now find the generated project files in the build-folder, which you process with your compiler environment.

(Feedback from developers on Windows on the build process of the developer version are welcome)


System requirements for the developer version:

* Boost libraries >= 1.45 (feel free to disable `BUILD_TESTING` and `ENABLE_UBLAS` in CMake in order to build without Boost)
* CMake 2.8 or higher (for building the tests and examples)
* A not-too-ancient C++ compiler

Optional:

* OpenMP-enabled C++ compiler
* One or more OpenCL SDKs (1.1 or higher)
* CUDA toolkit (4.x or higher)
* Eigen (3.0 or higher)
* MTL 4


Sending Pull Requests
--------------------------

We strive for high code quality and maintainability. Before sending pull requests, please ensure that your contribution passes the following minimum requirements, which are commonly considered good practice in software engineering:

* The new code passes all tests when running 'make test'.
* The new code compiles cleanly at high warning levels (-Wall -pedantic, just enable `ENABLE_PEDANTIC_FLAGS` within CMake) on at least GCC and/or Clang, ideally also Visual Studio compilers. The more the better, but at least one of the compilers should have been tested.
* For new functions or classes, please add Doxygen comments to the code. This makes it easier for others to quickly build on top of your code.
* Don't use tabs. Configure your editor such that it uses two spaces instead.

Thanks! :-)

