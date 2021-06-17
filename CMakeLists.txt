cmake_minimum_required(VERSION 3.12.1)
project(altro VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Werror)

# Add Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()
enable_testing()
include(GoogleTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include_directories(src/)

add_subdirectory(src)

add_subdirectory(examples)

add_subdirectory(test)
