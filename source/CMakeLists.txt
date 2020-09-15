set(Adamantine_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeam.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/NewtonSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Operator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Physics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.templates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/types.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/HeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ScanPath.hh
  )
set(Adamantine_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeam.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ImplicitOperator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/NewtonSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/HeatSource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ScanPath.cc
  )

if (ADAMANTINE_ENABLE_CUDA)
  set(Adamantine_HEADERS
      ${Adamantine_HEADERS}
      ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDevice.hh
      )
  set(Adamantine_SOURCES
      ${Adamantine_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDevice.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.cu
      )
endif()

# Because the Adamantine library is just used to simplify testing, we make it
# static. Thus, once the application is created it can be moved around. The
# other libraries can still be shared.
add_library(Adamantine STATIC ${Adamantine_SOURCES})

DEAL_II_SETUP_TARGET(Adamantine)

set_target_properties(Adamantine PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if (ADAMANTINE_ENABLE_CUDA)
set_target_properties(Adamantine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_compile_definitions(Adamantine PRIVATE ADAMANTINE_HAVE_CUDA)
endif()

target_link_libraries(Adamantine Boost::boost)
target_link_libraries(Adamantine Boost::chrono)
target_link_libraries(Adamantine Boost::filesystem)
target_link_libraries(Adamantine Boost::program_options)
target_link_libraries(Adamantine MPI::MPI_CXX)
target_include_directories(Adamantine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
