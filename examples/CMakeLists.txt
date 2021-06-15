include_directories(${PROJECT_SOURCE_DIR}/examples)

add_library(cost_function_examples
  quadratic_cost.cpp
)
target_link_libraries(cost_function_examples
  altro
  utils
)

add_library(dynamics_examples
  triple_integrator.cpp
)
target_link_libraries(dynamics_examples
  altro
)

add_subdirectory(test)