add_library(problem
  costfunction.cpp
  costfunction.hpp
  discretized_model.cpp
  discretized_model.hpp
  dynamics.cpp
  dynamics.hpp
  integration.cpp
  integration.hpp
  problem.cpp
  problem.hpp
)
target_link_libraries(problem
  PUBLIC
  Eigen3::Eigen
  fmt::fmt

  PRIVATE 
  common
  constraints
)
export_library(problem altroTargets)
