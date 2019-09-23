find_program(LCOV_EXECUTABLE lcov)
if(LCOV_EXECUTABLE)
  message(STATUS "Found lcov: ${LCOV_EXECUTABLE}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
else()
  message(SEND_ERROR "lcov not found")
endif()

find_program(GENHTML_EXECUTABLE genhtml)
if(GENHTML_EXECUTABLE)
  message(STATUS "Found genhtml: ${GENHTML_EXECUTABLE}")
else()
  message(SEND_ERROR "genhtml not found")
endif()

set(CPP_COVERAGE_FILE ${CMAKE_BINARY_DIR}/lcov.info)
set(CPP_COVERAGE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/htmlcov-cpp)

add_custom_target(coverage
  COMMAND ${LCOV_EXECUTABLE} 
  --capture
  --directory ${CMAKE_BINARY_DIR}
  --output-file=${CPP_COVERAGE_FILE}
  COMMAND ${GENHTML_EXECUTABLE}
  ${CPP_COVERAGE_FILE}
  --output-directory
  ${CPP_COVERAGE_OUTPUT_DIRECTORY}
  )
