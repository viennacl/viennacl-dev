
include(CTest)
include(CMakeDependentOption)
include(AddCCompilerFlagIfSupported)
include(AddCLinkerFlagIfSupported)

# Installation directories
##########################

set(INSTALL_INCLUDE_DIR include CACHE PATH
   "Installation directory for headers")
if(WIN32 AND NOT CYGWIN)
   set(DEF_INSTALL_CMAKE_DIR CMake)
else()
   set(DEF_INSTALL_CMAKE_DIR lib/cmake/viennacl)
endif()
set(INSTALL_CMAKE_DIR "${DEF_INSTALL_CMAKE_DIR}" CACHE PATH
   "Installation directory for CMake files")

if(NOT IS_ABSOLUTE "${INSTALL_CMAKE_DIR}")
   set(INSTALL_CMAKE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_CMAKE_DIR}")
endif()
file(RELATIVE_PATH CONF_REL_INSTALL_PREFIX "${INSTALL_CMAKE_DIR}"
   "${CMAKE_INSTALL_PREFIX}")
if(NOT IS_ABSOLUTE "${INSTALL_INCLUDE_DIR}")
   set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_INCLUDE_DIR}")
endif()
file(RELATIVE_PATH CONF_REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

# User options
##############

option(ENABLE_CUDA "Use the CUDA backend" OFF)

option(BUILD_EXAMPLES "Build example programs" ON)

option(ENABLE_OPENCL "Use the OpenCL backend" ON)

option(ENABLE_OPENMP "Use OpenMP acceleration" OFF)

option(ENABLE_ASAN "Build with address sanitizer if available" OFF)



# If you want to build the examples that use boost::numeric::ublas, enable
# the following:
cmake_dependent_option(ENABLE_UBLAS "Enable examples using uBLAS" OFF
   BUILD_EXAMPLES OFF)

# If you want to build the examples that use Armadillo
cmake_dependent_option(ENABLE_ARMADILLO "Enable examples that use Armadillo" OFF
   BUILD_EXAMPLES OFF)

# If you want to build the examples that use Eigen
cmake_dependent_option(ENABLE_EIGEN "Enable examples that use Eigen" OFF
   BUILD_EXAMPLES OFF)

# If you want to build the examples that use MTL4
cmake_dependent_option(ENABLE_MTL4 "Enable examples that use MTL4" OFF
   BUILD_EXAMPLES OFF)

option(ENABLE_PEDANTIC_FLAGS "Enable pedantic compiler flags (GCC and Clang only)" OFF)

mark_as_advanced(BOOSTPATH ENABLE_ASAN ENABLE_ARMADILLO ENABLE_EIGEN ENABLE_MTL4 ENABLE_PEDANTIC_FLAGS)

# Find prerequisites
####################

# Boost:
IF (BOOSTPATH)
 SET(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH}" "${BOOSTPATH}")
 SET(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH}" "${BOOSTPATH}/lib")
 SET(BOOST_ROOT "${BOOSTPATH}")
ENDIF (BOOSTPATH)


if(ENABLE_UBLAS OR BUILD_TESTING)
   set(Boost_USE_MULTITHREADED TRUE)
   find_package(Boost)
   if (Boost_MINOR_VERSION LESS 34)
     find_package(Boost REQUIRED COMPONENTS thread)
   elseif (Boost_MINOR_VERSION LESS 47)
     find_package(Boost REQUIRED COMPONENTS date_time serialization system thread)
   else ()
     find_package(Boost REQUIRED COMPONENTS chrono date_time serialization system thread)
   endif()
endif()

if (ENABLE_CUDA)
   find_package(CUDA REQUIRED)
   set(CUDA_ARCH_FLAG "-arch=sm_20" CACHE STRING "Use one out of sm_13, sm_20, sm_30, ...")
   set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${CUDA_ARCH_FLAG}" "-DVIENNACL_WITH_CUDA")
endif(ENABLE_CUDA)

if (ENABLE_OPENCL)
   find_package(OpenCL REQUIRED)
endif(ENABLE_OPENCL)

if (ENABLE_OPENMP)
   find_package(OpenMP REQUIRED)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DVIENNACL_WITH_OPENMP")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DVIENNACL_WITH_OPENMP")
   set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    ${OpenMP_EXE_LINKER_FLAGS}")
   set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_MODULE_LINKER_FLAGS}")
   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
   set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${OpenMP_STATIC_LINKER_FLAGS}")
endif(ENABLE_OPENMP)

if (ENABLE_ASAN)
  add_c_compiler_flag_if_supported("-fsanitize=address")
  add_c_linker_flag_if_supported("-fsanitize=address")
endif(ENABLE_ASAN)


if(ENABLE_ARMADILLO)
   # find Armadillo
   find_path(ARMADILLO_INCLUDE_DIR armadillo)
   if(NOT ARMADILLO_INCLUDE_DIR)
      message(SEND_ERROR "Failed to find Armadillo")
   endif()
   mark_as_advanced(ARMADILLO_INCLUDE_DIR)
endif()


if(ENABLE_EIGEN)
   # find Eigen
   find_path(EIGEN_INCLUDE_DIR Eigen/Dense)
   if(NOT EIGEN_INCLUDE_DIR)
      message(SEND_ERROR "Failed to find Eigen")
   endif()
   mark_as_advanced(EIGEN_INCLUDE_DIR)
endif()

if(ENABLE_MTL4)
   # MTL4 comes with a MTLConfig.cmake
   find_package(MTL REQUIRED)
endif()

if (ENABLE_OPENCL)
  include_directories(
   "${PROJECT_SOURCE_DIR}"
   ${OPENCL_INCLUDE_DIRS})
else (ENABLE_OPENCL)
  include_directories("${PROJECT_SOURCE_DIR}")
endif(ENABLE_OPENCL)

# Set high warning level on GCC
if(ENABLE_PEDANTIC_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wconversion")
endif()

# Disable Warning 4996 (std::copy is unsafe ...) on Visual Studio
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
endif()


# Export
########

configure_file(cmake/FindOpenCL.cmake
   "${PROJECT_BINARY_DIR}/FindOpenCL.cmake" COPYONLY)

configure_file(cmake/ViennaCLConfig.cmake.in
   "${PROJECT_BINARY_DIR}/ViennaCLConfig.cmake" @ONLY)

configure_file(cmake/ViennaCLConfigVersion.cmake.in
   "${PROJECT_BINARY_DIR}/ViennaCLConfigVersion.cmake" @ONLY)

if (CMAKE_MINOR_VERSION GREATER 6)  # export(PACKAGE ...) introduced with CMake 2.8.0
  export(PACKAGE ViennaCL)
endif()

# Install
#########

install(FILES
   "${PROJECT_BINARY_DIR}/FindOpenCL.cmake"
   "${PROJECT_BINARY_DIR}/ViennaCLConfig.cmake"
   "${PROJECT_BINARY_DIR}/ViennaCLConfigVersion.cmake"
   DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# For out-of-the-box support on MacOS:
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  IF (DARWIN_VERSION GREATER 12)
    IF (ENABLE_CUDA)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")  # Mavericks and beyond need the new C++ STL with CUDA
      # see https://github.com/viennacl/viennacl-dev/issues/106 for discussion
    ENDIF()
  ENDIF()
  INCLUDE_DIRECTORIES("/opt/local/include")
  SET(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
  set(CMAKE_MACOSX_RPATH 1) # Required for newer versions of CMake on MacOS X: http://www.kitware.com/blog/home/post/510
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
