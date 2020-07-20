cmake_minimum_required(VERSION 3.12)
project(gnns4code_native)

set(CMAKE_CXX_STANDARD 14)
set(Clang_DIR /usr/lib/llvm-10/lib/cmake/clang)

find_package(LLVM 10)
find_package(Clang)

set_target_properties(LLVM PROPERTIES
    IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C;CXX"
    IMPORTED_LOCATION_RELWITHDEBINFO "-lpthread"
)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.5.0
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

add_subdirectory(compy/representations/extractors)
