# - Find the headers and libraries required by ViennaProfiler
#
# Defines the following if found:
#  VIENNAPROFILER_FOUND        : TRUE if found, FALSE otherwise
#  VIENNAPROFILER_INCLUDE_DIRS : Include directories for MySQL++
#  VIENNAPROFILER_LIBRARIES    : The libraries to link against

# first find MySQL
find_path(MYSQL_INCLUDE_DIR mysql.h PATH_SUFFIXES mysql)
find_library(MYSQL_LIBRARY mysqlclient)

# now find MySQL++
if(MYSQL_INCLUDE_DIR AND MYSQL_LIBRARY)
   get_filename_component(_MYSQLPPROOT "${MYSQL_INCLUDE_DIR}" PATH)
   find_path(MYSQLPP_INCLUDE_DIR mysql++/mysql++.h
      HINTS "${_MYSQLPPROOT}/include")
   find_library(MYSQLPP_LIBRARY mysqlpp
      HINTS "${_MYSQLPPROOT}/lib")
endif()

# then find ViennaProfiler
find_path(VIENNAPROFILER_INCLUDE_DIR viennaprofiler/profiler.hpp)

mark_as_advanced(MYSQL_INCLUDE_DIR MYSQL_LIBRARY MYSQLPP_INCLUDE_DIR
   MYSQLPP_LIBRARY VIENNAPROFILER_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ViennaProfiler VIENNAPROFILER_INCLUDE_DIR
   MYSQLPP_INCLUDE_DIR MYSQLPP_LIBRARY MYSQL_INCLUDE_DIR MYSQL_LIBRARY)

set(VIENNAPROFILER_INCLUDE_DIRS "${VIENNAPROFILER_INCLUDE_DIR}"
   "${MYSQL_INCLUDE_DIR}" "${MYSQLPP_INCLUDE_DIR}")
set(VIENNAPROFILER_LIBRARIES "${MYSQL_LIBRARY}" "${MYSQLPP_LIBRARY}")
