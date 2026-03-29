set(Adamantine_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/BeamHeatSourceProperties.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BodyForce.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Boundary.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/CubeHeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/DataAssimilator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeamHeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ExperimentalData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/GaussianHeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/GoldakHeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/HeatSource.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialProperty.templates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialStates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MechanicalOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/MechanicalPhysics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Microstructure.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/PointCloud.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Quaternion.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/RayTracing.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ScanPath.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorBase.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperator.templates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDevice.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDevice.templates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInterface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysics.templates.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_management.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental_data_utils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/material_deposition.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/types.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/validate_input_database.hh
  )
set(Adamantine_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/BodyForce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Boundary.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CubeHeatSource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/DataAssimilator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ElectronBeamHeatSource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/GaussianHeatSource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/GoldakHeatSource.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialPropertyInstDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MaterialPropertyInstHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MechanicalOperator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/MechanicalPhysics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Microstructure.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/PointCloud.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Quaternion.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/RayTracing.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ScanPath.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorInstSHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorInstSLHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorInstSLPHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDeviceInstSDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDeviceInstSLDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalOperatorDeviceInstSLPDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSLDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSLPDev.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSLHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ThermalPhysicsInstSLPHost.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ensemble_management.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental_data_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/material_deposition.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/validate_input_database.cc
  )

add_library(${PROJECT_NAME} OBJECT)
target_sources(${PROJECT_NAME} PRIVATE ${Adamantine_SOURCES})
target_sources(${PROJECT_NAME} PUBLIC FILE_SET HEADERS FILES ${Adamantine_HEADERS})

DEAL_II_SETUP_TARGET(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_link_libraries(${PROJECT_NAME} Boost::boost)
target_link_libraries(${PROJECT_NAME} Boost::chrono)
target_link_libraries(${PROJECT_NAME} Boost::program_options)
target_link_libraries(${PROJECT_NAME} MPI::MPI_CXX)
if (ADAMANTINE_ENABLE_CALIPER)
  target_link_libraries(${PROJECT_NAME} caliper)
endif()


