# DO NOT CUSTOMIZE THIS FILE DIRECTLY!
# You can provide overrides/customizations in the file
# dashboardCustomSettings.cmake.

# parse var=val options
foreach(var_val IN LISTS CTEST_SCRIPT_ARG)
   if(var_val MATCHES "([^=]+)=(.*)")
      set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
   else()
      message(FATAL_ERROR
         "Invalid argument ${var_val}. "
         "Arguments must be of the form <var>=<value>")
   endif()
endforeach()
foreach(var CTEST_BUILD_NAME DASHBOARD_TYPE CTEST_NIGHTLY_START_TIME)
   if(NOT DEFINED "${var}")
      message(FATAL_ERROR
         "The variable ${var} is not defined on the command line.")
   endif()
endforeach()

# convenience variables
set(_custom_settings_file
   "${CTEST_SCRIPT_DIRECTORY}/dashboardCustomSettings.cmake")
set(_initcache_file
   "${CTEST_SCRIPT_DIRECTORY}/dashboardInitialCache.cmake")
set(_custom_initcache_file
   "${CTEST_SCRIPT_DIRECTORY}/dashboardCustomInitialCache.cmake")

# basic default settings
set(CTEST_SITE "$ENV{HOST}")
get_filename_component(CTEST_SOURCE_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}" PATH)
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/dashboard-build")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_CONFIGURATION_TYPE Release)
set(CTEST_COMMAND "/usr/bin/ctest")
#set(CTEST_UPDATE_COMMAND "/usr/bin/git")

# include the settings customization file
include("${_custom_settings_file}" OPTIONAL)

if(NOT EXISTS "${_initcache_file}")
   message(FATAL_ERROR "${_initcache_file} does not exist.")
endif()

set(CTEST_NOTES_FILES
   "${CTEST_SCRIPT_DIRECTORY}/${CTEST_SCRIPT_NAME}"
   "${_initcache_file}")
if(EXISTS "${_custom_initcache_file}")
   list(APPEND CTEST_NOTES_FILES "${_custom_initcache_file}")
endif()

set(CTEST_USE_LAUNCHERS TRUE)

# Want en_US ASCII output from tools
set(ENV{LC_ALL} C)

while(1)
   ctest_empty_binary_directory("${CTEST_BINARY_DIRECTORY}")
   ctest_start("${DASHBOARD_TYPE}")

   if(CTEST_UPDATE_COMMAND AND NOT DASHBOARD_TYPE STREQUAL "Experimental")
      ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}" RETURN_VALUE result)
   endif()

   # Only proceed if we updated a file or if we're not a continuous build
   set(BUILD_AND_TEST FALSE)
   if(NOT DASHBOARD_TYPE STREQUAL "Continuous")
      message(STATUS "Not a continuous dashboard, will build and test.")
      set(BUILD_AND_TEST TRUE)
   elseif(result GREATER 0)
      message(STATUS "One or more files were updated, will build and test.")
      set(BUILD_AND_TEST TRUE)
   endif()

   if(BUILD_AND_TEST)
      ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE result
         OPTIONS "-DCTEST_BUILD_NAME=${CTEST_BUILD_NAME};-C${_initcache_file}")
      if(result EQUAL 0)
         # Only try to build if we could configure
         ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE result)
      endif()
      if(result EQUAL 0)
         # Only try to test if we could build
         ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE result)
      endif()
      # Always submit results
      ctest_submit()
   endif()
   if(DASHBOARD_TYPE STREQUAL "Continuous")
      ctest_sleep(300)
   else()
      break()
   endif()
endwhile()
