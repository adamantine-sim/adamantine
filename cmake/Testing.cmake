function(adamantine_ADD_BOOST_TEST TEST_NAME)
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cc ${tests_SOURCES})
    target_include_directories(${TEST_NAME} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_CHRONO_LIBRARY})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_FILESYSTEM_LIBRARY})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_MPI_LIBRARY})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_PROGRAM_OPTIONS_LIBRARY})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_TIMER_LIBRARY})
    target_link_libraries(${TEST_NAME} PUBLIC ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    target_include_directories(${TEST_NAME} SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} PUBLIC ${DEAL_II_LIBRARIES})
    target_link_libraries(${TEST_NAME} LINK_PUBLIC Adamantine)
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
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

function(adamantine_ADD_INTEGRATION_TEST TEST_NAME)
    if(ARGN)
        set(NUMBER_OF_PROCESSES_TO_EXECUTE ${ARGN})
    else()
        set(NUMBER_OF_PROCESSES_TO_EXECUTE 1)
    endif()
    foreach(NPROC ${NUMBER_OF_PROCESSES_TO_EXECUTE})
        add_test(
            NAME ${TEST_NAME}_${NPROC}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${NPROC}
                ${CMAKE_BINARY_DIR}/bin/adamantine --input-file=${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}
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
  string(REGEX REPLACE "/" "_" DUMMY ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE})
  add_custom_target(
    ${DUMMY} ALL
    DEPENDS ${CMAKE_BINARY_DIR}/bin/${INPUT_FILE}
    )
endfunction()
