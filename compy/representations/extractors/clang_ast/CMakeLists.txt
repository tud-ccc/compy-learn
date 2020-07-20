# Extractor library
add_library(clang_extractor
        clang_extractor.cc
        clang_graph_frontendaction.cc
        clang_seq_frontendaction.cc
        )
target_link_libraries(clang_extractor
        extractors_common
        )
target_compile_options(clang_extractor PRIVATE
        ${compile_options_common}
        )

# Extractor tests
add_executable(clang_extractor_tests
        clang_extractor_test.cc
        )
target_link_libraries(clang_extractor_tests
        clang_extractor

        gmock
        gtest
        gtest_main
        )
target_compile_options(clang_extractor_tests PRIVATE
        -fno-rtti -fPIC
        )