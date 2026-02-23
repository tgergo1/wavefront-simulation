cmake_minimum_required(VERSION 3.24)

project(wavefront-simulation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WAVEFRONT_BUILD_TESTS "Build unit/integration tests" ON)
option(WAVEFRONT_BUILD_PYTHON "Build Python bindings" ON)
option(WAVEFRONT_ENABLE_EXACT "Enable exact-reference arithmetic support" ON)
option(WAVEFRONT_USE_FETCHCONTENT "Fetch pinned third-party dependencies" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

include(cmake/Dependencies.cmake)

set(WAVEFRONT_SOURCES
    src/api/problem_validation.cpp
    src/api/solver_factory.cpp
    src/core/runtime_solver.cpp
    src/physics/boundary.cpp
    src/physics/interface.cpp
    src/symbolic/expression.cpp
    src/exact/exact_number.cpp
)

add_library(wavefront ${WAVEFRONT_SOURCES})
add_library(wavefront::wavefront ALIAS wavefront)

target_include_directories(
    wavefront
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(wavefront PUBLIC Threads::Threads)

if(WAVEFRONT_ENABLE_EXACT AND WAVEFRONT_LIMITLESS_INCLUDE_DIR)
  target_include_directories(wavefront PRIVATE ${WAVEFRONT_LIMITLESS_INCLUDE_DIR})
  target_compile_definitions(wavefront PUBLIC WAVEFRONT_HAS_LIMITLESS=1)
else()
  target_compile_definitions(wavefront PUBLIC WAVEFRONT_HAS_LIMITLESS=0)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(wavefront PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(
    TARGETS wavefront
    EXPORT wavefrontTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    EXPORT wavefrontTargets
    NAMESPACE wavefront::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wavefront
)

configure_package_config_file(
    cmake/WavefrontConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/WavefrontConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wavefront
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/WavefrontConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/WavefrontConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/WavefrontConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wavefront
)

if(WAVEFRONT_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(WAVEFRONT_BUILD_PYTHON)
  add_subdirectory(python)
endif()
