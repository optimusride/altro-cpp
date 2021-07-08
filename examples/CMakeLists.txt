add_library(cost_function_examples
  quadratic_cost.cpp
  quadratic_cost.hpp
)
target_link_libraries(cost_function_examples
  problem
  utils
)

add_library(dynamics_examples
  unicycle.hpp
  unicycle.cpp
  triple_integrator.cpp
  triple_integrator.hpp
)
target_link_libraries(dynamics_examples
  problem
  utils
)

add_library(constraint_examples
  basic_constraints.cpp
  basic_constraints.hpp
)
target_link_libraries(constraint_examples
  constraints
  utils
)