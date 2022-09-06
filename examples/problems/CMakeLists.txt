add_library(example_problems
  triple_integrator.hpp
  triple_integrator.cpp
  unicycle.hpp
  unicycle.cpp
)
target_link_libraries(example_problems
  PUBLIC
  altro
  constraint_examples
  cost_function_examples
  dynamics_examples
)

