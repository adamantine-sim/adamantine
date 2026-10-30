cmake_minimum_required(VERSION 3.15)

if(POLICY CMP0167)
  # Policy introduced in CMake 3.30. Use the Boost CMake Config installed with Boost instead of
  # the FindBoost.cmake module installed with CMake.
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# This needs to be called before project is called and before we setup the other
# TPL and adamantine
include(SetupDealII)

project(Adamantine LANGUAGES CXX VERSION 1.1.9)

find_package(Git QUIET)
if (Git_FOUND)
  if (EXISTS "${CMAKE_SOURCE_DIR}/.git/HEAD")
    # Re-run CMake when HEAD changes, including when a branch ref is moved.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${CMAKE_SOURCE_DIR}/.git/HEAD"
    )
    file(READ "${CMAKE_SOURCE_DIR}/.git/HEAD" ADAMANTINE_GIT_HEAD_CONTENT
      LIMIT 256
    )
    if (ADAMANTINE_GIT_HEAD_CONTENT MATCHES "ref: ([^\n]+)")
      set(ADAMANTINE_GIT_REF "${CMAKE_MATCH_1}")
      set(ADAMANTINE_GIT_REF_FILE "${CMAKE_SOURCE_DIR}/.git/${ADAMANTINE_GIT_REF}")
      if (EXISTS "${ADAMANTINE_GIT_REF_FILE}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
          "${ADAMANTINE_GIT_REF_FILE}"
        )
      endif()
    endif()
  endif()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ADAMANTINE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE ADAMANTINE_GIT_RESULT
  )
endif()
if (NOT Git_FOUND OR NOT ADAMANTINE_GIT_RESULT EQUAL 0 OR NOT ADAMANTINE_GIT_SHA)
  set(ADAMANTINE_GIT_SHA "unknown")
endif()

include(SetupTPLs)
include(SetupAdamantine)

add_subdirectory(applications)
add_subdirectory(source)

option(ADAMANTINE_ENABLE_COVERAGE "Measure coverage" OFF)
if (ADAMANTINE_ENABLE_COVERAGE)
  include(CodeCoverage)
endif()

option(ADAMANTINE_ENABLE_TESTS "Build tests" OFF)
if (ADAMANTINE_ENABLE_TESTS)
  include(Testing)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# Provide "indent" target for indenting all the header and the source files.
add_custom_target(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./indent
)

option(ADAMANTINE_ENABLE_DOCUMENTATION "Build documentation" OFF)
if (ADAMANTINE_ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()
