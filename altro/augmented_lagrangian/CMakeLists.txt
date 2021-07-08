add_library(augmented_lagrangian
	al_cost.hpp
	al_cost.cpp
)
target_link_libraries(augmented_lagrangian
	constraints
	problem
)