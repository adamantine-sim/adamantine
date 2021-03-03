function(adamantine_ADD_BOOST_TEST TEST_NAME)
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cc ${tests_SOURCES})
    DEAL_II_SETUP_TARGET(${TEST_NAME})
    target_link_libraries(${TEST_NAME} Boost::boost)
    target_link_libraries(${TEST_NAME} Boost::chrono)
    target_link_libraries(${TEST_NAME} Boost::filesystem)
    target_link_libraries(${TEST_NAME} Boost::program_options)
    target_link_libraries(${TEST_NAME} Boost::timer)
    target_link_libraries(${TEST_NAME} Boost::unit_test_framework)
    target_link_libraries(${TEST_NAME} MPI::MPI_CXX)
    target_link_libraries(${TEST_NAME} Adamantine)
    if (ADAMANTINE_ENABLE_CUDA)
        target_compile_definitions(${TEST_NAME} PRIVATE ADAMANTINE_HAVE_CUDA)
    endif()
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    if(ARGN)
        set(NUMBER_OF_PROCESSES_TO_EXECUTE ${ARGN})
    else()
        set(NUMBER_OF_PROCESSES_TO_EXECUTE 1)
    endif()
    foreach(NPROC ${NUMBER_OF_PROCESSES_TO_EXECUTE})
        add_test(
            NAME ${TEST_NAME}_${NPROC}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NPROC} ${CMAKE_BINARY_DIR}/bin/${TEST_NAME}
        )
        set_tests_properties(${TEST_NAME}_${NPROC} PROPERTIES
            PROCESSORS ${NPROC}
        )
    endforeach()
endfunction()

function(adamantine_ADD_BOOST_CUDA_TEST TEST_NAME)
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cu ${tests_SOURCES})
    DEAL_II_SETUP_TARGET(${TEST_NAME})
    target_link_libraries(${TEST_NAME} Boost::boost)
    target_link_libraries(${TEST_NAME} Boost::chrono)
    target_link_libraries(${TEST_NAME} Boost::filesystem)
    target_link_libraries(${TEST_NAME} Boost::program_options)
    target_link_libraries(${TEST_NAME} Boost::timer)
    target_link_libraries(${TEST_NAME} Boost::unit_test_framework)
    target_link_libraries(${TEST_NAME} MPI::MPI_CXX)
    target_link_libraries(${TEST_NAME} Adamantine)
    target_compile_definitions(${TEST_NAME} PRIVATE ADAMANTINE_HAVE_CUDA)
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if(ARGN)
        set(NUMBER_OF_PROCESSES_TO_EXECUTE ${ARGN})
    else()
        set(NUMBER_OF_PROCESSES_TO_EXECUTE 1)
    endif()
    foreach(NPROC ${NUMBER_OF_PROCESSES_TO_EXECUTE})
        add_test(
            NAME ${TEST_NAME}_${NPROC}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NPROC} ${CMAKE_BINARY_DIR}/bin/${TEST_NAME}
        )
        set_tests_properties(${TEST_NAME}_${NPROC} PROPERTIES
            PROCESSORS ${NPROC}
        )
    endforeach()
endfunction()

function(adamantine_COPY_INPUT_FILE INPUT_FILE PATH_TO_FILE)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE}
    DEPENDS ${CMAKE_SOURCE_DIR}/${PATH_TO_FILE}/${INPUT_FILE}
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy ${CMAKE_SOURCE_DIR}/${PATH_TO_FILE}/${INPUT_FILE} ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE}
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy ${CMAKE_SOURCE_DIR}/${PATH_TO_FILE}/${INPUT_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${INPUT_FILE}
    COMMENT "Copying ${INPUT_FILE}"
    )
  string(REGEX REPLACE "[/@]" "_" DUMMY ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE})
  add_custom_target(
    ${DUMMY} ALL
    DEPENDS ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE}
    )
endfunction()
