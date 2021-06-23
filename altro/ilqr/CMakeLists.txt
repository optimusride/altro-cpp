add_library(ilqr
	cost_expansion.cpp
	cost_expansion.hpp
	dynamics_expansion.cpp
	dynamics_expansion.hpp
	knot_point_function_type.cpp
	knot_point_function_type.hpp
	ilqr.cpp
	ilqr.hpp
)
target_link_libraries(ilqr
	common
	problem
	utils
	Eigen3::Eigen
)
export_library(ilqr altroTargets "cost_expansion.hpp;ilqr.hpp;knot_point_function_type.hpp;dynamics_expansion.hpp" altro/ilqr)

add_subdirectory(test)
