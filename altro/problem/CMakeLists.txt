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
  common
  constraints
)
export_library(problem altroTargets "costfunction.hpp;discretized_model.hpp;dynamics.hpp;integration.hpp;problem.hpp" altro/problem)
