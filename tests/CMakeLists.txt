include(${CMAKE_SOURCE_DIR}/cmake/Testing.cmake)

add_definitions(-DBOOST_TEST_DYN_LINK)

include_directories(${CMAKE_SOURCE_DIR}/source)

set(UNIT_TESTS "")
list(APPEND
     UNIT_TESTS
     test_data_assimilator
     test_experimental_data
     test_geometry
     test_heat_source
     test_integration_3d_device
     test_material_property
     test_material_property_device
     test_mechanical_operator
     test_mechanical_physics
     test_post_processor
     test_quaternion
     test_scan_path
     test_thermal_operator
     test_thermal_operator_device
     test_thermal_physics_device
     test_timer
     test_utils
     test_validate_input_database
     )

set(MPI_UNIT_TESTS "")
list(APPEND
     MPI_UNIT_TESTS
     test_integration_2d
     test_integration_2d_device
     test_integration_3d
     test_integration_3d_amr
     test_integration_3d_amr_device
     test_integration_da_augmented
     test_integration_thermoelastic
     test_material_deposition
     test_microstructure
     test_thermal_physics
     test_ensemble_management
    )

# test_integration_da is a special test that we need to test with more
# processors than the others
adamantine_ADD_BOOST_TEST(test_integration_da 1 2 3 6)

foreach(TEST_NAME ${UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME})
endforeach()

foreach(TEST_NAME ${MPI_UNIT_TESTS})
  adamantine_ADD_BOOST_TEST(${TEST_NAME} 1 2)
endforeach()


adamantine_COPY_DATA_FILE(amr_test.info)
adamantine_COPY_DATA_FILE(analytical_1D_transient.py)
adamantine_COPY_DATA_FILE(bare_plate_L_da.info)
adamantine_COPY_DATA_FILE(bare_plate_L_da_aug_ref_data_0_0.csv)
adamantine_COPY_DATA_FILE(bare_plate_L_da_aug_ref_data_0_1.csv)
adamantine_COPY_DATA_FILE(bare_plate_L_da_aug_ref_log.txt)
adamantine_COPY_DATA_FILE(bare_plate_L_da_augmented.info)
adamantine_COPY_DATA_FILE(bare_plate_L_ensemble.info)
adamantine_COPY_DATA_FILE(bare_plate_L_expt_data_0_0.csv)
adamantine_COPY_DATA_FILE(bare_plate_L_expt_data_0_1.csv)
adamantine_COPY_DATA_FILE(bare_plate_L_expt_log.txt)
adamantine_COPY_DATA_FILE(bare_plate_L_scan_path.txt)
adamantine_COPY_DATA_FILE(bare_plate_L_scan_path_fast.txt)
adamantine_COPY_DATA_FILE(convective_bc_gold.txt)
adamantine_COPY_DATA_FILE(demo_316_short.info)
adamantine_COPY_DATA_FILE(demo_316_short_amr.info)
adamantine_COPY_DATA_FILE(demo_316_short_anisotropic.info)
adamantine_COPY_DATA_FILE(demo_316_short_scan_path.txt)
adamantine_COPY_DATA_FILE(demo_316_short_scan_path_amr.txt)
adamantine_COPY_DATA_FILE(experiment_log_test.txt)
adamantine_COPY_DATA_FILE(experimental_data_0_0.csv)
adamantine_COPY_DATA_FILE(extruded_cube.msh)
adamantine_COPY_DATA_FILE(hexahedra.msh)
adamantine_COPY_DATA_FILE(HourGlass_AOP.info)
adamantine_COPY_DATA_FILE(HourGlass_AOP.vtk)
adamantine_COPY_DATA_FILE(HourGlass_AOP_scan_path.txt)
adamantine_COPY_DATA_FILE(integration_2d.info)
adamantine_COPY_DATA_FILE(integration_2d_ensemble.info)
adamantine_COPY_DATA_FILE(integration_2d_gold.txt)
adamantine_COPY_DATA_FILE(integration_2d_units.info)
adamantine_COPY_DATA_FILE(integration_3d_gold.txt)
adamantine_COPY_DATA_FILE(integration_3d_gold_0.txt)
adamantine_COPY_DATA_FILE(integration_3d_gold_1.txt)
adamantine_COPY_DATA_FILE(integration_3d_gold_short.txt)
adamantine_COPY_DATA_FILE(integration_3d_gold_short_0.txt)
adamantine_COPY_DATA_FILE(integration_3d_gold_short_1.txt)
adamantine_COPY_DATA_FILE(integration_da_add_material.info)
adamantine_COPY_DATA_FILE(integration_da_add_material_expt_log.txt)
adamantine_COPY_DATA_FILE(integration_da_add_material_expt_pc_0_0.csv)
adamantine_COPY_DATA_FILE(integration_da_add_material_expt_ray_0_0.csv)
adamantine_COPY_DATA_FILE(integration_da_add_material_sp.txt)
adamantine_COPY_DATA_FILE(material_deposition_2d.txt)
adamantine_COPY_DATA_FILE(material_deposition_3d.txt)
adamantine_COPY_DATA_FILE(material_path_test_material_deposition.txt)
adamantine_COPY_DATA_FILE(material_property_polynomial.info)
adamantine_COPY_DATA_FILE(material_property_table.info)
adamantine_COPY_DATA_FILE(microstructure_G_R_gold_1.txt)
adamantine_COPY_DATA_FILE(microstructure_G_R_gold_2.txt)
adamantine_COPY_DATA_FILE(rays_cam-0-0_test_full.csv)
adamantine_COPY_DATA_FILE(raytracing_experimental_data_0_0.csv)
adamantine_COPY_DATA_FILE(raytracing_non_AA_cells-0-0.csv)
adamantine_COPY_DATA_FILE(scan_path.txt)
adamantine_COPY_DATA_FILE(scan_path_5_axis.txt)
adamantine_COPY_DATA_FILE(scan_path_diagonal.txt)
adamantine_COPY_DATA_FILE(scan_path_event_series.inp)
adamantine_COPY_DATA_FILE(scan_path_L.txt)
adamantine_COPY_DATA_FILE(scan_path_layers.txt)
adamantine_COPY_DATA_FILE(scan_path_test_thermal_physics.txt)
adamantine_COPY_DATA_FILE(scan_path_units.txt)
adamantine_COPY_DATA_FILE(thermoelastic_bare_plate.info)
adamantine_COPY_DATA_FILE(thermoelastic_bare_plate_add_material_scan_path.txt)
adamantine_COPY_DATA_FILE(thermoelastic_bare_plate_scan_path.txt)
