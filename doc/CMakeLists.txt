if(BUILD_DOXYGEN_DOCS)
   configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
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

if(BUILD_MANUAL)
   set(MANUAL_SRCS
      manual/algorithms.tex
      manual/benchmarks.tex
      manual/changelogs.tex
      manual/contributors.tex
      manual/cover.tex
      manual/custom-contexts.tex
      manual/custom-kernels.tex
      manual/design.tex
      manual/figures/logo_px200.eps
      manual/figures/matvec2.eps
      manual/figures/note.eps
      manual/figures/solver.eps
      manual/figures/tip.eps
      manual/figures/TU_Signet_CMYK.eps
      manual/IEEEtran_v1.13.bst
      manual/installation.tex
      manual/introduction.tex
      manual/keywords.tex
      manual/license.tex
      manual/multi-device.tex
      manual/operations.tex
      manual/other-libs.tex
      manual/setup.tex
      manual/tuning.tex
      manual/types.tex
      manual/versioning.tex
      manual/viennacl.bib
      manual/viennacl.tex)

   set(MANUAL_OBJS
      "${CMAKE_CURRENT_BINARY_DIR}/viennacl.pdf"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.bbl"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.blg"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.dvi"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.idx"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.log"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.out"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.ps"
      "${CMAKE_CURRENT_BINARY_DIR}/manual/viennacl.toc")
   foreach(f IN LISTS MANUAL_SRCS)
      configure_file(${f} ${CMAKE_CURRENT_BINARY_DIR}/${f} COPYONLY)
      if(f MATCHES "(.*)\\.tex$")
         list(APPEND MANUAL_OBJS
            "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_MATCH_1}.aux")
      endif()
   endforeach()

   add_custom_command(OUTPUT ${MANUAL_OBJS}
      COMMAND ${LATEX_COMPILER} viennacl.tex
      COMMAND ${LATEX_COMPILER} viennacl.tex
      COMMAND ${BIBTEX_COMPILER} viennacl
      COMMAND ${LATEX_COMPILER} viennacl.tex
      COMMAND ${BIBTEX_COMPILER} viennacl
      COMMAND ${LATEX_COMPILER} viennacl.tex
      COMMAND ${DVIPS_CONVERTER} -Ppdf -G0 -ta4 viennacl.dvi
      COMMAND ${PS2PDF_CONVERTER} -dPDFSETTINGS=/prepress
         -dCompatibilityLevel=1.3 -dMAxSubsetPct=100 -dSubsetFonts=true
         -dEmbedAllFonts=true -sPAPERSIZE=a4 -dAutoRotatePages=/None
         -dOptimize=true viennacl.ps ../viennacl.pdf
      DEPENDS ${MANUAL_SRCS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/manual
      COMMENT "Generating manual viennacl.pdf"
      VERBATIM)

   add_custom_target(manual ALL
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/viennacl.pdf")
endif()
