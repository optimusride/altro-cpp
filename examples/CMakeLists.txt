add_library(cost_function_examples
  quadratic_cost.cpp
  quadratic_cost.hpp
)
target_link_libraries(cost_function_examples
  PUBLIC
  problem

  PRIVATE
  utils
)

add_library(dynamics_examples
  unicycle.hpp
  unicycle.cpp
  triple_integrator.cpp
  triple_integrator.hpp
)
target_link_libraries(dynamics_examples
  PUBLIC
  problem

  PRIVATE
  utils
)

add_library(constraint_examples
  basic_constraints.cpp
  basic_constraints.hpp
  obstacle_constraints.cpp
  obstacle_constraints.hpp
)
target_link_libraries(constraint_examples
  PUBLIC
  constraints

  PRIVATE
  utils
)

add_subdirectory(problems)