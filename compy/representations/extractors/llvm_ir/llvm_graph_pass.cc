#include "llvm_graph_pass.h"

#include <iostream>
#include <utility>

#include "llvm/Analysis/CallGraph.h"

#include "llvm_graph_funcinfo.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace graph {

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo());

  // Collect and dump all the function information
  for (auto &func : module.functions()) {
    // Skip functions without definition (fwd declarations)
    if (func.isDeclaration()) {
      continue;
    }

    auto &pass = getAnalysis<FunctionInfoPass>(func);
    auto functionInfo = std::move(pass.getInfo());
    info->functionInfos.push_back(std::move(functionInfo));
  }

  // Dump the call graph
  info->callGraphInfo.reset(new CallGraphInfo());

  const auto &callGraph = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  for (auto &kv : callGraph) {
    auto *func = kv.first;
    auto &node = kv.second;

    // Skip the null entry
    if (func == nullptr) continue;

    // -1, because the null entry references everything
    for (auto &kv : *node) {
      // Skip for functions without definition (fwd declarations)
      if (kv.second->getFunction()) {
        info->callGraphInfo->calls.push_back(
            kv.second->getFunction()->getName().str());
      }
    }
  }

  this->extractionInfo = info;

  // Returning false indicates that we didn't change anything
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<CallGraphWrapperPass>();
  au.addRequired<FunctionInfoPass>();
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("graphExtractor", "GraphExtractor",
                                             true /* Only looks at CFG */,
                                             true /* Analysis Pass */);

}  // namespace graph
}  // namespace llvm
}  // namespace compy
