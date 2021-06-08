add_library(trajectory
  trajectory.cpp
  integration.cpp
)
target_link_libraries(trajectory
  Eigen3::Eigen
)

add_library(altro
  dynamics.cpp
  costfunction.cpp
)
target_link_libraries(altro
  trajectory
  Eigen3::Eigen
)

add_subdirectory(test)