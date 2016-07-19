# Create adamantine executable and link against the static library.
add_executable(adamantine adamantine.cc)
target_link_libraries(adamantine LINK_PUBLIC Adamantine)
