# Generate Doxygen documentation
FIND_PACKAGE(Doxygen)

CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
)

ADD_CUSTOM_TARGET(documentation
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMENT "Generating API documentation with Doxygen"
)

# Generate the latex document with the equations and the algorithmic choices
FIND_PACKAGE(LATEX COMPONENTS PDFLATEX)
ADD_CUSTOM_TARGET(latex
    ${PDFLATEX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/adamantine.tex
    COMMENT "Generating Latex document."
)
