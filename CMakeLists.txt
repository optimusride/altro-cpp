cmake_minimum_required(VERSION 3.12.1)
list(APPEND CMAKE_MESSAGE_CONTEXT altro)
project(AltroCpp VERSION 0.3.4)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Wextra -pedantic -Werror)

# Defines GNU standard installation directories
include(GNUInstallDirs)
include(CMakePrintHelpers)
include(FindThreads)

# Check if MacOS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX TRUE)

    # Add the Homebrew headers for Big Sur
    include_directories( "/opt/homebrew/include/" )
endif()

cmake_policy(SET CMP0077 NEW)

# Add custom functions
include(${AltroCpp_SOURCE_DIR}/cmake/Functions.cmake)

################################################################################
############################### OPTIONS ########################################
################################################################################
option(ALTRO_RUN_CLANG_TIDY "Enable static analysis w/ clang tidy" OFF)
option(ALTRO_BUILD_TESTS "Build the tests" ON)
option(ALTRO_BUILD_EXAMPLES "Build the examples" ON)
option(ALTRO_BUILD_COVERAGE "Build test coverage" OFF)
option(ALTRO_BUILD_BENCHMARKS "Build the benchmark problems" ON)
option(ALTRO_BUILD_DOCS "Build the altro documentation" OFF)
option(ALTRO_SET_POSITION_INDEPENDENT "Set the -fPIC option on all targets" ON)
option(ALTRO_BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Handle default build type
set(ALTRO_DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Setting CMAKE_BUILD_TYPE to ${MOCAP_DEFAULT_BUILD_TYPE}")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set (and create) log file directory
set(LOGDIR ${PROJECT_BINARY_DIR}/logs CACHE FILESTRING "Location of log files")
file(MAKE_DIRECTORY ${LOGDIR})

# Examples are required to build the tests
if (ALTRO_BUILD_TESTS)
  set(ALTRO_BUILD_EXAMPLES ON)
endif()

set(BUILD_SHARED_LIBS ${ALTRO_BUILD_SHARED_LIBS})

################################################################################
############################# ADD DEPENDENCIES #################################
################################################################################
include(FetchContent)

# Add libfmt
list(APPEND CMAKE_MESSAGE_CONTEXT fmt)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG a33701196adfad74917046096bf5a2aa0ab0bb50  # version 9.1.0
)
FetchContent_MakeAvailable(fmt)
export_library(fmt AltroCppTargets)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

# Add Eigen
list(APPEND CMAKE_MESSAGE_CONTEXT eigen)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation." FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # version 3.4.0
)
FetchContent_MakeAvailable(eigen)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

################################################################################
########################## ENABLE GOOGLE TEST ##################################
################################################################################

# This flag is set to ensure that gtest and gmock files do not get installed along with rest of the package. 
set(INSTALL_GTEST off CACHE BOOL "" FORCE)

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
if (ALTRO_BUILD_TESTS)
  enable_testing()
endif()
include(GoogleTest)

################################################################################
############################ DOCUMENTATION #####################################
################################################################################

if(ALTRO_BUILD_DOCS)
  message(STATUS "Adding altro_docs target to build documentation.")
  add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

################################################################################
############################ CODE COVERAGE #####################################
################################################################################
# Code Coverage
# Build by:
# cmake -DCMAKE_BUILD_TYPE=Debug -DALTRO_BUILD_COVERAGE=ON ..
# Run: "make coverage" and open the build/altro_test_coverage.html
# Currently all .html and .css are littered in /build
# TODO(jw) : move all report files to a dedicated sub-directory


if(ALTRO_BUILD_COVERAGE)
    if (MACOSX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    else()
        add_compile_options(-O0 -g -fprofile-arcs -ftest-coverage)
        add_link_options(-fprofile-arcs -ftest-coverage)
    endif (MACOSX)

    message(STATUS "Building altro with code coverage")
#  add_custom_target(coverage ALL
#      COMMAND gcovr -r ../ --html-details --exclude-directories='_deps' -o altro_test_coverage.html
#  )

endif()


################################################################################
############################# BUILD STEPS ######################################
################################################################################

# Output all binaries to a single folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Make all include paths relative to the root folder
include_directories(${PROJECT_SOURCE_DIR})

# This adds the -fPIC option for all build targets.
# Recommended for integration into other repos.
set(CMAKE_POSITION_INDEPENDENT_CODE ${ALTRO_SET_POSITION_INDEPENDENT})

add_subdirectory(altro)
if (ALTRO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if (ALTRO_BUILD_TESTS)
  add_subdirectory(test)
endif()
if (ALTRO_BUILD_BENCHMARKS)
  add_subdirectory(perf)
endif()

add_subdirectory(dev/brian)


################################################################################
############################# INSTALLATION #####################################
################################################################################
# NOTE: The following steps allow installing this project for external use.
# These steps should be done from the outer-most cmake file only.

# Override default install location to be in `~/.local`
# This is the place where the package headers and libraries will get installed.
# This same prefix is also used by `find_package` cmake module.
# This can be changed by setting the `CMAKE_INSTALL_PREFIX` cache variable.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "~/.local" CACHE PATH "..." FORCE)
endif()

# Export the AltroCppTargets.cmake file
install(EXPORT AltroCppTargets
  FILE AltroCppTargets.cmake
  NAMESPACE altro::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AltroCpp
)

# Generate the package config file that includes the exported targets
include(CMakePackageConfigHelpers)
configure_package_config_file(${AltroCpp_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/AltroCppConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AltroCpp
)

# Generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/AltroCppConfigVersion.cmake"
  VERSION "${AltroCpp_VERSION_MAJOR}.${AltroCpp_VERSION_MINOR}.${AltroCpp_VERSION_PATCH}"
  COMPATIBILITY AnyNewerVersion
)

# Install all of the headers, keeping the source directory
# Note that leaving off the backslash on DIRECTORY adds `altro` as the top-level
# directory under the `include` folder.
install(DIRECTORY ${AltroCpp_SOURCE_DIR}/altro # No trailing backslash here
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp" # filter all header files
)

# Install the configuration file and the version file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/AltroCppConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/AltroCppConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AltroCpp
)
