cmake_minimum_required(VERSION 3.16)
project(cann_composed_probe LANGUAGES C CXX)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory(../../../ggml ggml)
add_executable(cann-composed probe.cpp)
target_link_libraries(cann-composed PRIVATE ggml ggml-cpu)
target_compile_features(cann-composed PRIVATE cxx_std_17)
if(TARGET ggml-cann)
    # Static ggml-cann propagates SDK library names, but its PRIVATE search
    # directories do not reach executable consumers. Reuse the backend's
    # configured directories without hardcoding an SDK installation path.
    target_link_directories(cann-composed PRIVATE
        "$<TARGET_PROPERTY:ggml-cann,LINK_DIRECTORIES>")
endif()
