# Common compile options
set(compile_options_common
        -fno-rtti -fPIC
        -Wall -Wextra
        # LLVM warnings
        -Wno-unused-parameter -Wno-tautological-overlap-compare -Wno-deprecated-copy -Wno-class-memaccess -Wno-maybe-uninitialized
        # Clang warnings
        -Wno-comment -Wno-strict-aliasing)

# Common library
add_library(extractors_common
        common/clang_driver.cc)
llvm_map_components_to_libnames(REQ_LLVM_LIBRARIES ${LLVM_TARGETS_TO_BUILD}
        asmparser
        core
        linker
        bitreader
        irreader
        ipo
        scalaropts
        analysis
        support
        frontendopenmp
        option
        passes
        objcarcopts
        coroutines
        lto
        coverage
        aarch64codegen
        )
target_include_directories(extractors_common PUBLIC
        .
        ${LLVM_INCLUDE_DIRS}
        ${CLANG_INCLUDE_DIRS}
        )
target_link_libraries(extractors_common
        -Wl,--start-group
        ${REQ_LLVM_LIBRARIES}
        clangBasic
        clangFrontendTool
        -Wl,--end-group
        )
target_compile_options(extractors_common PRIVATE
        ${compile_options_common}
        )

# Common tests
add_executable(extractors_common_tests
        common/clang_driver_test.cc
        )
target_link_libraries(extractors_common_tests
        extractors_common

        gmock
        gtest
        gtest_main
        )
target_compile_options(extractors_common_tests PRIVATE
        -fno-rtti -fPIC
        )

# LLVM pybind11 module
pybind11_add_module(extractors
        extractors.cc
        )
target_link_libraries(extractors PRIVATE
        clang_extractor
        llvm_extractor
        )
target_compile_options(extractors PRIVATE
        -Wno-unused-value
        )

add_subdirectory(clang_ast)
add_subdirectory(llvm_ir)