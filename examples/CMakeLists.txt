add_library(cost_function_examples
  quadratic_cost.cpp
  quadratic_cost.hpp
)
target_link_libraries(cost_function_examples
  problem
  utils
)

add_library(dynamics_examples
  triple_integrator.cpp
  triple_integrator.hpp
)
target_link_libraries(dynamics_examples
  problem
  utils
)