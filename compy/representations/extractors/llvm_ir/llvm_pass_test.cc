#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"

#include "common/common_test.h"
#include "gtest/gtest.h"
#include "llvm_graph_pass.h"
#include "llvm_seq_pass.h"

using namespace llvm;
using namespace compy;
using namespace compy::llvm;

using std::string;

class LLVMGraphPassFixture : public testing::Test {
 protected:
  void SetUp() override {
    // Register other llvm passes
    PassRegistry& reg = *PassRegistry::getPassRegistry();
    initializeCallGraphWrapperPassPass(reg);
    initializeMemorySSAWrapperPassPass(reg);

    // Setup the pass manager, add pass
    _pm = new legacy::PassManager();
    _ep = new graph::ExtractorPass();
    _pm->add(_ep);
  }

  void TearDown() override {
    free(_pm);
    free(_ep);
  }

  graph::ExtractionInfoPtr Extract(std::string ir) {
    // Construct an IR file from the filename passed on the command line.
    SMDiagnostic err;
    LLVMContext context;
    MemoryBufferRef mb = MemoryBuffer::getMemBuffer(ir)->getMemBufferRef();
    std::unique_ptr<Module> module = parseIR(mb, err, context);
    if (!module.get()) {
      throw std::runtime_error("Failed compiling to LLVM module");
    }

    // Run pass
    _pm->run(*module);

    // Return extraction info
    return _ep->extractionInfo;
  }

  legacy::PassManager* _pm;
  graph::ExtractorPass* _ep;
};

class LLVMSeqPassFixture : public testing::Test {
 protected:
  void SetUp() override {
    // Register other llvm passes
    PassRegistry& reg = *PassRegistry::getPassRegistry();
    initializeCallGraphWrapperPassPass(reg);
    initializeMemorySSAWrapperPassPass(reg);

    // Setup the pass manager, add pass
    _pm = new legacy::PassManager();
    _ep = new seq::ExtractorPass();
    _pm->add(_ep);
  }

  void TearDown() override {
    free(_pm);
    free(_ep);
  }

  seq::ExtractionInfoPtr Extract(std::string ir) {
    // Construct an IR file from the filename passed on the command line.
    SMDiagnostic err;
    LLVMContext context;
    MemoryBufferRef mb = MemoryBuffer::getMemBuffer(ir)->getMemBufferRef();
    std::unique_ptr<Module> module = parseIR(mb, err, context);
    if (!module.get()) {
      throw std::runtime_error("Failed compiling to LLVM module");
    }

    // Run pass
    _pm->run(*module);

    // Return extraction info
    return _ep->extractionInfo;
  }

  legacy::PassManager* _pm;
  seq::ExtractorPass* _ep;
};

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveSuccess) {
  graph::ExtractionInfoPtr info = Extract(kLLVM1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);
}

TEST_F(LLVMSeqPassFixture, RunPassAndRetrieveSuccess2) {
  seq::ExtractionInfoPtr info = Extract(kLLVM1);

  ASSERT_EQ(info->functionInfos.size(), 1UL);

  std::vector<std::string> signature = info->functionInfos[0]->signature;

  seq::BasicBlockInfoPtr basicBlock = info->functionInfos[0]->basicBlocks[0];
  ASSERT_GT(basicBlock->instructions.size(), 1UL);

  seq::InstructionInfoPtr instructionInfoPtr = basicBlock->instructions[0];
  ASSERT_GT(instructionInfoPtr->tokens.size(), 1UL);
}

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveFail) {
  EXPECT_THROW(
      {
        try {
          graph::ExtractionInfoPtr info = Extract(kLLVM2);
        } catch (std::runtime_error const& err) {
          EXPECT_EQ(err.what(), std::string("Failed compiling to LLVM module"));
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(LLVMGraphPassFixture, RunPassAndRetrieveZero) {
  graph::ExtractionInfoPtr info = Extract("");

  ASSERT_EQ(info->functionInfos.size(), 0UL);
}
