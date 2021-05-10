#pragma once

#include <memory>
#include <string>
#include <vector>

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "llvm_extractor.h"

namespace compy {
namespace llvm {
namespace seq {

class ExtractorPass : public ::llvm::ModulePass {
 public:
  static char ID;
  ExtractorPass() : ::llvm::ModulePass(ID) {}

  bool runOnModule(::llvm::Module &M) override;
  void getAnalysisUsage(::llvm::AnalysisUsage &au) const override;

  ExtractionInfoPtr extractionInfo;
};

}  // namespace seq
}  // namespace llvm
}  // namespace compy
