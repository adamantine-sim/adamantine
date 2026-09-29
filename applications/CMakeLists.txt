# Create adamantine executable and link against the static library.
set(Adamantine_app_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/adamantine.hh
)
set(Adamantine_app_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/adamantine.cc
)
add_executable(adamantine ${Adamantine_app_SOURCES} ${Adamantine_app_HEADERS})

set_target_properties(adamantine PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

DEAL_II_SETUP_TARGET(adamantine)
target_link_libraries(adamantine Adamantine)
if (ADAMANTINE_ENABLE_ADIAK)
  target_link_libraries(adamantine adiak::adiak)
endif()

# Create an executable that visualizes the cells activated by the scan path.
add_executable(activation_visualizer activation_visualizer.cc)

set_target_properties(activation_visualizer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

DEAL_II_SETUP_TARGET(activation_visualizer)
target_link_libraries(activation_visualizer Adamantine)

# Install executables
install(TARGETS adamantine activation_visualizer)

# Copy input files
file(COPY input.info DESTINATION ${CMAKE_BINARY_DIR}/bin)
file(COPY input_scan_path.txt DESTINATION ${CMAKE_BINARY_DIR}/bin)
