set(WAVEFRONT_DOCTEST_TARGET "")
if(TARGET doctest::doctest)
  set(WAVEFRONT_DOCTEST_TARGET doctest::doctest)
elseif(TARGET doctest)
  set(WAVEFRONT_DOCTEST_TARGET doctest)
endif()

if(NOT WAVEFRONT_DOCTEST_TARGET)
  message(FATAL_ERROR "doctest target not found")
endif()

set(WAVEFRONT_TEST_SOURCES
    main.cpp
    unit/test_parser.cpp
    unit/test_factory.cpp
    unit/test_boundary_interface.cpp
    integration/test_mode_parity.cpp
    verification/test_invariants.cpp
    verification/test_convergence.cpp
    verification/test_physical_validity.cpp
    benchmarks/test_published_cases.cpp
)

add_executable(wavefront_tests ${WAVEFRONT_TEST_SOURCES})
target_link_libraries(wavefront_tests PRIVATE wavefront ${WAVEFRONT_DOCTEST_TARGET})
target_compile_features(wavefront_tests PRIVATE cxx_std_20)

add_test(NAME wavefront_tests COMMAND wavefront_tests)
