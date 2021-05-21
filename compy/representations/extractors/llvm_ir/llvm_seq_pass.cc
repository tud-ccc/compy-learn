#include "llvm_seq_pass.h"

#include <algorithm>
#include <string>

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/Transforms/IPO.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace seq {

class InfoBuilder {
 public:
  InfoBuilder() {
    functionInfo_.reset(new FunctionInfo);
    largestSlotIdSoFar_ = 0;
  }

  void AddToken(std::string token) { tokenBuffer_.push_back(token); }

  void onBasicBlockStart() {
    // Before first BB was the function signature.
    if (functionInfo_->basicBlocks.empty()) {
      stripComments(tokenBuffer_);
      stripNewlines(tokenBuffer_);
      stripEntryInstruction(tokenBuffer_);

      functionInfo_->signature = tokenBuffer_;
      tokenBuffer_.clear();
    }

    BasicBlockInfoPtr basicBlockInfo(new BasicBlockInfo);
    basicBlockInfo->name = std::to_string(++largestSlotIdSoFar_);

    functionInfo_->basicBlocks.push_back(basicBlockInfo);
  }

  void onInstructionStart() { tokenBuffer_.clear(); }

  void onInstructionEnd() {
    stripDoubleWhitespaces(tokenBuffer_);

    InstructionInfoPtr instructionInfo(new InstructionInfo);
    instructionInfo->tokens = tokenBuffer_;

    BasicBlockInfoPtr basicBlockInfo = functionInfo_->basicBlocks.back();
    basicBlockInfo->instructions.push_back(instructionInfo);

    // Track largest slot id.
    for (std::size_t i = 0; i != instructionInfo->tokens.size() - 2; ++i) {
      if (instructionInfo->tokens[i] == "%" &&
          instructionInfo->tokens[i + 2] == " = ") {
        int slotId = std::stoi(instructionInfo->tokens[i + 1]);
        largestSlotIdSoFar_ = std::max(largestSlotIdSoFar_, slotId);
      }
    }
  }

  FunctionInfoPtr getInfo() { return functionInfo_; }

  FunctionInfoPtr functionInfo_;

 private:
  void stripComments(std::vector<std::string> &tokens) {
    std::vector<std::string>::iterator semicolonEle;

    bool semicolonFound = false;
    for (auto it = tokens.begin(); it != tokens.end(); it++) {
      auto token = *it;

      if (token.find(";") != std::string::npos) {
        semicolonEle = it;
        semicolonFound = true;
      } else if (semicolonFound && token.find("\n") != std::string::npos) {
        tokens.erase(semicolonEle, it);
        semicolonFound = false;
      }
    }
  }

  void stripEntryInstruction(std::vector<std::string> &tokens) {
    auto itEntry = std::find(tokens.begin(), tokens.end(), "entry");
    tokens.erase(itEntry - 1, tokens.end());
  }

  void stripNewlines(std::vector<std::string> &tokens) {
    tokens.erase(std::remove(tokens.begin(), tokens.end(), "\n"), tokens.end());
  }

  void stripDoubleWhitespaces(std::vector<std::string> &tokens) {
    tokens.erase(std::remove(tokens.begin(), tokens.end(), "  "), tokens.end());
  }

 private:
  int largestSlotIdSoFar_;
  std::vector<std::string> tokenBuffer_;
};
using InfoBuilderPtr = std::shared_ptr<InfoBuilder>;

class token_ostream : public ::llvm::raw_ostream {
  void write_impl(const char *Ptr, size_t Size) override {
    std::string str(Ptr, Size);
    infoBuilder_->AddToken(str);

    OS.append(Ptr, Size);
  }

  uint64_t current_pos() const override { return OS.size(); }

 public:
  explicit token_ostream(InfoBuilderPtr infoBuilder)
      : infoBuilder_(infoBuilder) {
    // Set unbufferd, so we get token-by-token
    SetUnbuffered();
  }
  ~token_ostream() override { flush(); }

  std::string &str() {
    flush();
    return OS;
  }

  std::string &getStr() { return OS; }

 private:
  InfoBuilderPtr infoBuilder_;
  std::string OS;
};

class TokenAnnotator : public ::llvm::AssemblyAnnotationWriter {
 public:
  TokenAnnotator(InfoBuilderPtr infoBuilder) : infoBuilder_(infoBuilder) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *bb,
                                        formatted_raw_ostream &) {
    infoBuilder_->onBasicBlockStart();
  }
  virtual void emitInstructionAnnot(const Instruction *,
                                    formatted_raw_ostream &) {
    infoBuilder_->onInstructionStart();
  }
  virtual void printInfoComment(const Value &, formatted_raw_ostream &) {
    infoBuilder_->onInstructionEnd();
  }

 private:
  InfoBuilderPtr infoBuilder_;
};

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  for (const auto &F : module.functions()) {
    // InfoBuilder holds the state of the tokenization. It is built using a
    // custom stream that captures token by token. An Annotator object with hook
    // functions is regularly called by the LLVM stack, structuring the token
    // stream into the entities.
    InfoBuilderPtr infoBuilder(new InfoBuilder);

    token_ostream TokenStream(infoBuilder);
    std::unique_ptr<AssemblyAnnotationWriter> tokenAnnotator(
        new TokenAnnotator(infoBuilder));

    F.print(TokenStream, tokenAnnotator.get());

    FunctionInfoPtr functionInfo = infoBuilder->getInfo();
    functionInfo->name = F.getName().str();
    functionInfo->str = TokenStream.getStr();
    info->functionInfos.push_back(functionInfo);
  }

  this->extractionInfo = info;

  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<CallGraphWrapperPass>();

  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("seqExtractor", "SeqExtractor",
                                             true /* Only looks at CFG */,
                                             true /* Analysis Pass */);

}  // namespace seq
}  // namespace llvm
}  // namespace compy
