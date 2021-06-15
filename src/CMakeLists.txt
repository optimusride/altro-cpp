add_subdirectory(utils)

add_library(trajectory
  trajectory.cpp
)
target_link_libraries(trajectory
  Eigen3::Eigen
)

add_library(altro
  dynamics.cpp
  costfunction.cpp
  discretized_model.cpp
  integration.cpp
)
target_link_libraries(altro
  trajectory
  Eigen3::Eigen
  utils
)

add_subdirectory(test)