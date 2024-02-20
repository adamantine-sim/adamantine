# Create adamantine executable and link against the static library.
add_executable(adamantine adamantine.cc)
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

file(COPY input.info DESTINATION ${CMAKE_BINARY_DIR}/bin)
file(COPY input_scan_path.txt DESTINATION ${CMAKE_BINARY_DIR}/bin)
