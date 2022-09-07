add_executable(cartpole
  cartpole_main.cpp
  cartpole.cpp cartpole.hpp)
target_link_libraries(cartpole
  PRIVATE
  altro
  cost_function_examples
)
