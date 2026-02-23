include(FetchContent)

find_package(Threads REQUIRED)

set(WAVEFRONT_LIMITLESS_INCLUDE_DIR "" CACHE PATH "Path to limitless headers")

if(WAVEFRONT_ENABLE_EXACT)
  if(NOT WAVEFRONT_LIMITLESS_INCLUDE_DIR)
    find_path(
      WAVEFRONT_LIMITLESS_INCLUDE_DIR
      NAMES limitless.hpp
      DOC "Directory containing limitless.hpp"
    )
  endif()

  if(NOT WAVEFRONT_LIMITLESS_INCLUDE_DIR AND WAVEFRONT_USE_FETCHCONTENT)
    FetchContent_Declare(
      limitless
      GIT_REPOSITORY https://github.com/tgergo1/limitless.git
      GIT_TAG 41af45dcba41985d6e88bd6dbd30b8d4bdbe2f18
      GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(limitless)
    if(NOT limitless_POPULATED)
      if(POLICY CMP0169)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_Populate(limitless)
      if(POLICY CMP0169)
        cmake_policy(POP)
      endif()
    endif()
    set(WAVEFRONT_LIMITLESS_INCLUDE_DIR "${limitless_SOURCE_DIR}" CACHE PATH "Path to limitless headers" FORCE)
  endif()

  if(NOT WAVEFRONT_LIMITLESS_INCLUDE_DIR)
    message(FATAL_ERROR "WAVEFRONT_ENABLE_EXACT=ON requires limitless.hpp. Set WAVEFRONT_LIMITLESS_INCLUDE_DIR or WAVEFRONT_USE_FETCHCONTENT=ON")
  endif()
endif()

if(WAVEFRONT_BUILD_TESTS)
  find_package(doctest CONFIG QUIET)
  if(NOT doctest_FOUND AND WAVEFRONT_USE_FETCHCONTENT)
    set(DOCTEST_NO_INSTALL ON CACHE BOOL "Disable doctest install rules" FORCE)
    FetchContent_Declare(
      doctest
      GIT_REPOSITORY https://github.com/doctest/doctest.git
      GIT_TAG v2.4.11
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(doctest)
  endif()
endif()

if(WAVEFRONT_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND AND WAVEFRONT_USE_FETCHCONTENT)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  if(NOT pybind11_FOUND AND NOT TARGET pybind11::module)
    message(FATAL_ERROR "WAVEFRONT_BUILD_PYTHON=ON requires pybind11. Install it or set WAVEFRONT_USE_FETCHCONTENT=ON.")
  endif()
endif()
