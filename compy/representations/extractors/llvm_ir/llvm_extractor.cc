#include "llvm_extractor.h"

#include <string>

#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"

#include "llvm_graph_pass.h"
#include "llvm_seq_pass.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace llvm {

LLVMIRExtractor::LLVMIRExtractor(ClangDriverPtr clangDriver)
    : clangDriver_(clangDriver) {}

graph::ExtractionInfoPtr LLVMIRExtractor::GraphFromString(std::string src) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  passes.push_back(createStripSymbolsPass());

  graph::ExtractorPass *extractorPass = new graph::ExtractorPass();
  passes.push_back(extractorPass);

  clangDriver_->Invoke(src, frontendActions, passes);

  return extractorPass->extractionInfo;
}

seq::ExtractionInfoPtr LLVMIRExtractor::SeqFromString(std::string src) {
  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  passes.push_back(createStripSymbolsPass());
  seq::ExtractorPass *pass = new seq::ExtractorPass();
  passes.push_back(pass);

  clangDriver_->Invoke(src, frontendActions, passes);

  return pass->extractionInfo;
}

}  // namespace llvm
}  // namespace compy
