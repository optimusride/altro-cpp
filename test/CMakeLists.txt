# Add examples to include path
include_directories(${PROJECT_SOURCE_DIR}/examples)

add_subdirectory(common)
add_subdirectory(problem)
add_subdirectory(utils)
add_subdirectory(ilqr)
add_subdirectory(constraints)
add_subdirectory(augmented_lagrangian)
