include(CheckCXXCompilerFlag)
macro(add_c_linker_flag_if_supported FLAG)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}")
  CHECK_CXX_COMPILER_FLAG(${FLAG} FLAG_SUPPORTED)
  if(FLAG_SUPPORTED)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${FLAG}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${FLAG}")
  endif()
endmacro(add_c_linker_flag_if_supported)


