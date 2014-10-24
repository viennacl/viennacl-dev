#SET(MTL_INCLUDE_DIRS "${MTL_DIR}/../../include")
find_package(Boost 1.36 REQUIRED)
if(Boost_FOUND)
  LIST(APPEND MTL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

# find MTL
find_path(MTL_INCLUDE_DIR boost/numeric/itl)
if(NOT MTL_INCLUDE_DIR)
  message(SEND_ERROR "Failed to find MTL")
endif()
mark_as_advanced(MTL_INCLUDE_DIR)

include_directories(${MTL_INCLUDE_DIRS} "${MTL_INCLUDE_DIR}")
