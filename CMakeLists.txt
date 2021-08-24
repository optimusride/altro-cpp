cmake_minimum_required(VERSION 3.12.1)
project(altro VERSION 0.3.2)

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

# Add custom functions
include(${altro_SOURCE_DIR}/cmake/Functions.cmake)

################################################################################
############################### OPTIONS ########################################
################################################################################
option(ALTRO_RUN_CLANG_TIDY "Enable static analysis w/ clang tidy" OFF)
option(ALTRO_BUILD_TESTS "Build the tests" ON)
option(ALTRO_BUILD_EXAMPLES "Build the examples" ON)
option(ALTRO_BUILD_COVERAGE "Build test coverage" OFF)
option(ALTRO_BUILD_BENCHMARKS "Build the benchmark problems" ON)
option(ALTRO_SET_POSITION_INDEPENDENT "Set the -fPIC option on all targets" ON)
option(ALTRO_BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Set (and create) log file directory
set(LOGDIR "${PROJECT_BINARY_DIR}/logs" CACHE FILESTRING "Location of log files")
file(MAKE_DIRECTORY ${LOGDIR})

# Examples are required to build the tests
if (ALTRO_BUILD_TESTS)
  set(ALTRO_BUILD_EXAMPLES ON)
endif()

set(BUILD_SHARED_LIBS ${ALTRO_BUILD_SHARED_LIBS})

################################################################################
############################# ADD DEPENDENCIES #################################
################################################################################
# Add eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# NOTE(bjackson): Can we use a newer version? Should we bundle it with the 
# source code?
set(fmt_ROOT "/opt/fmt")
set(fmt_REQUIRED_VERSION 6.2.1)
set(fmt_REQUIRED_DEBIAN_VERSION ${fmt_REQUIRED_VERSION})
find_package(fmt ${fmt_REQUIRED_VERSION} REQUIRED)

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
if (${BUILD_TEST})
  enable_testing()
endif()
include(GoogleTest)

################################################################################
############################ DOCUMENTATION #####################################
################################################################################

add_subdirectory(docs)

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
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    endif (MACOSX)

add_custom_target(coverage ALL
    COMMAND gcovr -r ../ --html-details --exclude-directories='_deps' -o altro_test_coverage.html
)

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

# Export the altroTargets.cmake file
install(EXPORT altroTargets
  FILE altroTargets.cmake
  NAMESPACE altro::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)

# Generate the package config file that includes the exported targets
include(CMakePackageConfigHelpers)
configure_package_config_file(${altro_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)

# Generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake"
  VERSION "${altro_VERSION_MAJOR}.${altro_VERSION_MINOR}.${altro_VERSION_PATCH}"
  COMPATIBILITY AnyNewerVersion
)

# Install all of the headers, keeping the source directory
# Note that leaving off the backslash on DIRECTORY adds `altro` as the top-level
# directory under the `include` folder.
install(DIRECTORY ${altro_SOURCE_DIR}/altro # No trailing backslash here
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp" # filter all header files
)

# Install the configuration file and the version file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/altro
)