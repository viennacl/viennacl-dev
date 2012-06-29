SET(MTL_INCLUDE_DIRS "${MTL_DIR}/../../include")
find_package(Boost 1.36 REQUIRED)
if(Boost_FOUND)
	LIST(APPEND MTL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

include_directories(${MTL_INCLUDE_DIRS})
