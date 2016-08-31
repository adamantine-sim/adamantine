set(Adamantine_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeam.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Operator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/types.hh
  )
set(Adamantine_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeam.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitOperator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.cc
  )

# Because the Adamantine library is just used to simplify testing, we make it
# static. Thus, once the application is created it can be moved around. The
# other libraries can still be shared.
add_library(Adamantine STATIC ${Adamantine_SOURCES})

set_target_properties(Adamantine PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(Adamantine SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_link_libraries(Adamantine PUBLIC ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(Adamantine PUBLIC ${Boost_MPI_LIBRARY})
target_link_libraries(Adamantine PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_include_directories(Adamantine SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(Adamantine PUBLIC ${DEAL_II_LIBRARIES})
target_include_directories(Adamantine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
