#include "clang_extractor.h"

#include <string>

#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"

#include "clang_graph_frontendaction.h"
#include "clang_seq_frontendaction.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace clang {

ClangExtractor::ClangExtractor(ClangDriverPtr clangDriver)
    : clangDriver_(clangDriver) {}

graph::ExtractionInfoPtr ClangExtractor::GraphFromString(std::string src) {
  auto fa = std::make_unique<compy::clang::graph::ExtractorFrontendAction>();

  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  frontendActions.push_back(fa.get());

  clangDriver_->Invoke(src, frontendActions, passes);

  return fa->extractionInfo;
}

seq::ExtractionInfoPtr ClangExtractor::SeqFromString(std::string src) {
  auto fa = std::make_unique<compy::clang::seq::ExtractorFrontendAction>();

  std::vector<::clang::FrontendAction *> frontendActions;
  std::vector<::llvm::Pass *> passes;

  frontendActions.push_back(fa.get());

  clangDriver_->Invoke(src, frontendActions, passes);

  return fa->extractionInfo;
}

}  // namespace clang
}  // namespace compy
