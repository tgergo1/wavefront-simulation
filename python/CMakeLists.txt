if(NOT TARGET pybind11::module)
  message(FATAL_ERROR "pybind11::module target is required for Python bindings")
endif()

pybind11_add_module(_wavefront MODULE bindings.cpp)

target_link_libraries(_wavefront PRIVATE wavefront)
target_compile_features(_wavefront PRIVATE cxx_std_20)

set_target_properties(
  _wavefront
  PROPERTIES
    OUTPUT_NAME "_wavefront"
)

install(TARGETS _wavefront LIBRARY DESTINATION wavefront)
install(FILES wavefront/__init__.py DESTINATION wavefront)
