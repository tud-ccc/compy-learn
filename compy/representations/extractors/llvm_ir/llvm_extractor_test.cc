#include "llvm_extractor.h"

#include <fstream>
#include <iostream>

#include "common/common_test.h"
#include "gtest/gtest.h"

#define TO_STRING(prefix) #prefix
#define COMPILER_BINARY(prefix) TO_STRING(prefix) "/bin/clang"

using namespace ::llvm;
using namespace compy;
using namespace compy::llvm;

using LE = LLVMIRExtractor;
using CD = ClangDriver;

constexpr char kProgram4ForwardDecl[] = "int barbara(float x, float y);";

void createFileWithContents(std::string filename, std::string filecontent) {
  std::ofstream tempHeaderFile(filename.c_str());
  tempHeaderFile << filecontent << std::endl;
  tempHeaderFile.close();
}
void removeFile(std::string filename) { std::remove(filename.c_str()); }

class LLVMExtractorFixture : public testing::Test {
 protected:
  void Init(CD::ProgrammingLanguage programmingLanguage) {
    // Init extractor
    std::vector<std::tuple<std::string, CD::IncludeDirType>> includeDirs = {};
    std::vector<std::string> compilerFlags = {"-Werror"};

    driver_.reset(new ClangDriver(programmingLanguage,
                                  CD::OptimizationLevel::O0, includeDirs,
                                  compilerFlags));
    driver_->setCompilerBinary(COMPILER_BINARY(CLANG_INSTALL_PREFIX));
    extractor_.reset(new LE(driver_));
  }

  std::shared_ptr<CD> driver_;
  std::shared_ptr<LE> extractor_;
};

class LLVMExtractorCFixture : public LLVMExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::C); }
};

class LLVMExtractorCPlusPlusFixture : public LLVMExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::CPLUSPLUS); }
};

class LLVMExtractorLLVMFixture : public LLVMExtractorFixture {
 protected:
  void SetUp() override { Init(CD::ProgrammingLanguage::LLVM); }
};

// C tests
TEST_F(LLVMExtractorCFixture, ExtractFromFunction1) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "foo");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunction2) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram2);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "max");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunction5) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram5);

  ASSERT_EQ(info->functionInfos.size(), 2UL);
  ASSERT_EQ(info->functionInfos[0]->name, "max");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunctionWithSystemInclude) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram3);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "foo");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromFunctionWithUserInclude) {
  std::string headerFilename = "/tmp/tempHdr.h";
  createFileWithContents(headerFilename, kProgram4ForwardDecl);

  driver_->addIncludeDir("/tmp", CD::IncludeDirType::SYSTEM);
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram4);

  removeFile(headerFilename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromNoFunction) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString("");

  ASSERT_EQ(info->functionInfos.size(), 0UL);
}

TEST_F(LLVMExtractorCFixture, ExtractFromBadFunction) {
  EXPECT_THROW(
      {
        try {
          graph::ExtractionInfoPtr info = extractor_->GraphFromString("foobar");
        } catch (std::runtime_error const& err) {
          EXPECT_EQ(err.what(), std::string("Failed compiling to LLVM module"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(LLVMExtractorCFixture, ExtractWithDifferentOptimizationlevels) {
  driver_->setOptimizationLevel(CD::OptimizationLevel::O0);
  graph::ExtractionInfoPtr infoO0 = extractor_->GraphFromString(kProgram2);

  driver_->setOptimizationLevel(CD::OptimizationLevel::O1);
  graph::ExtractionInfoPtr infoO1 = extractor_->GraphFromString(kProgram2);

  ASSERT_TRUE(infoO0->functionInfos[0]->basicBlocks.size() >
              infoO1->functionInfos[0]->basicBlocks.size());
}

// C++ tests
TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunction1) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3foov");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunction2) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram2);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3maxii");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 2UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunctionWithSystemInclude) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram3);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "_Z3foov");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

TEST_F(LLVMExtractorCPlusPlusFixture, ExtractFromFunctionWithUserInclude) {
  std::string headerFilename = "/tmp/tempHdr.h";
  createFileWithContents(headerFilename, kProgram4ForwardDecl);

  driver_->addIncludeDir("/tmp", CD::IncludeDirType::SYSTEM);
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kProgram4);

  removeFile(headerFilename);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->args.size(), 0UL);
}

// LLVM tests
TEST_F(LLVMExtractorLLVMFixture, ExtractFromFunction1) {
  graph::ExtractionInfoPtr info = extractor_->GraphFromString(kLLVM1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
  ASSERT_EQ(info->functionInfos[0]->name, "A");
  ASSERT_EQ(info->functionInfos[0]->args.size(), 1UL);
}
