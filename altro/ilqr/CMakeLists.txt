add_library(ilqr
	cost_expansion.cpp
	cost_expansion.hpp
	ilqr.cpp
	ilqr.hpp
)
target_link_libraries(ilqr
	common
	problem
	utils
	Eigen3::Eigen
)
export_library(ilqr altroTargets "cost_expansion.hpp;ilqr.hpp" altro/ilqr)

add_subdirectory(test)
