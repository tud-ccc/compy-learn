# Extractor library
add_library(llvm_extractor
        llvm_extractor.cc
        llvm_graph_pass.cc
        llvm_graph_funcinfo.cc
        llvm_seq_pass.cc
        )
target_link_libraries(llvm_extractor
        extractors_common
        )
target_compile_options(llvm_extractor PRIVATE
        ${compile_options_common}
        )

# Extractor tests
add_executable(llvm_extractor_tests
        llvm_pass_test.cc
        llvm_extractor_test.cc
        )
target_link_libraries(llvm_extractor_tests
        llvm_extractor

        gmock
        gtest
        gtest_main
        )
target_compile_options(llvm_extractor_tests PRIVATE
        -fno-rtti -fPIC
        )
target_compile_definitions(llvm_extractor_tests PRIVATE
        CLANG_INSTALL_PREFIX=${CLANG_INSTALL_PREFIX}
        )
