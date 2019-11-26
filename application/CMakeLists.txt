# Create adamantine executable and link against the static library.
add_executable(adamantine adamantine.cc)
DEAL_II_SETUP_TARGET(adamantine)
target_link_libraries(adamantine Adamantine)
file(COPY input.info DESTINATION ${CMAKE_BINARY_DIR}/bin)
