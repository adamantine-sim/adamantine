set(adamantine_HEADERS
  )
set(adamantine_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/adamantine.cc
  )

add_executable(adamantine ${adamantine_SOURCES})

set_target_properties(adamantine PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(adamantine SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(adamantine PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_include_directories(adamantine SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(adamantine PUBLIC ${DEAL_II_LIBRARIES})
