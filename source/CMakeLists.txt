set(adamantine_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Operator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/types.hh
  )
set(adamantine_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/adamantine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.cc
  )

add_executable(adamantine ${adamantine_SOURCES})

set_target_properties(adamantine PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(adamantine SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(adamantine PUBLIC ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(adamantine PUBLIC ${Boost_MPI_LIBRARY})
target_link_libraries(adamantine PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_include_directories(adamantine SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(adamantine PUBLIC ${DEAL_II_LIBRARIES})
