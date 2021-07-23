add_subdirectory(common)
add_subdirectory(ilqr)
add_subdirectory(problem)
add_subdirectory(utils)
add_subdirectory(constraints)
add_subdirectory(augmented_lagrangian)

add_library(altro
  main.cpp
  eigentypes.hpp
)

target_link_libraries(altro 
  augmented_lagrangian
  common
  constraints
  ilqr
  problem
  utils
)

export_library(altro altroTargets "eigentypes.hpp" altro)
