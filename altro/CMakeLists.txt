add_subdirectory(common)
add_subdirectory(ilqr)
add_subdirectory(problem)
add_subdirectory(utils)

add_library(altro
  main.cpp
  eigentypes.hpp
)
export_library(altro altroTargets "eigentypes.hpp" altro)
