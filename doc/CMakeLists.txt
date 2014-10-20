option(BUILD_DOXYGEN_DOCS "Build the doxygen documentation" FALSE)

if(BUILD_DOXYGEN_DOCS)
   find_package(Doxygen REQUIRED)

   configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
   configure_file(doxyfilter.py ${CMAKE_CURRENT_BINARY_DIR}/doxyfilter.py @ONLY)

   add_custom_command(OUTPUT
      ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/index.html
      COMMAND ${DOXYGEN_EXECUTABLE}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      VERBATIM)
   # Runs only once
   add_custom_target(apidoc-all ALL
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/index.html)
   # For manual updates
   add_custom_target(apidoc
      COMMAND ${DOXYGEN_EXECUTABLE}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      VERBATIM)
endif()

