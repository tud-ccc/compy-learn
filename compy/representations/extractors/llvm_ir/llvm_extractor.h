#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "common/clang_driver.h"
#include "common/visitor.h"

namespace compy {
namespace llvm {

namespace seq {
struct InstructionInfo;
using InstructionInfoPtr = std::shared_ptr<InstructionInfo>;

struct BasicBlockInfo;
using BasicBlockInfoPtr = std::shared_ptr<BasicBlockInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct InstructionInfo : IVisitee {
  std::vector<std::string> tokens;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct BasicBlockInfo : IVisitee {
  std::string name;
  std::vector<InstructionInfoPtr> instructions;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : instructions) it->accept(v);
  }
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<std::string> signature;
  std::vector<BasicBlockInfoPtr> basicBlocks;
  std::string str;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : basicBlocks) it->accept(v);
  }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : functionInfos) it->accept(v);
  }
};
}  // namespace seq

namespace graph {
struct OperandInfo;
using OperandInfoPtr = std::shared_ptr<OperandInfo>;

struct ArgInfo;
using ArgInfoPtr = std::shared_ptr<ArgInfo>;

struct ConstantInfo;
using ConstantInfoPtr = std::shared_ptr<ConstantInfo>;

struct InstructionInfo;
using InstructionInfoPtr = std::shared_ptr<InstructionInfo>;

struct BasicBlockInfo;
using BasicBlockInfoPtr = std::shared_ptr<BasicBlockInfo>;

struct MemoryAccessInfo;
using MemoryAccessInfoPtr = std::shared_ptr<MemoryAccessInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct CallGraphInfo;
using CallGraphInfoPtr = std::shared_ptr<CallGraphInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct OperandInfo : IVisitee {
  virtual ~OperandInfo() = default;
};

struct ArgInfo : OperandInfo {
  std::string name;
  std::string type;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct ConstantInfo : OperandInfo {
  std::string type;
  std::string value;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct InstructionInfo : OperandInfo {
  std::string type;
  std::string opcode;
  std::string callTarget;
  bool isLoadOrStore;
  std::vector<OperandInfoPtr> operands;
  FunctionInfoPtr function;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct BasicBlockInfo : IVisitee {
  std::string name;
  std::vector<InstructionInfoPtr> instructions;
  std::vector<BasicBlockInfoPtr> successors;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : instructions) it->accept(v);
  }
};

struct MemoryAccessInfo {
  std::string type;
  InstructionInfoPtr inst;
  BasicBlockInfoPtr block;
  std::vector<MemoryAccessInfoPtr> dependencies;
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::string type;
  InstructionInfoPtr entryInstruction;
  std::vector<InstructionInfoPtr> exitInstructions;
  std::vector<ArgInfoPtr> args;
  std::vector<BasicBlockInfoPtr> basicBlocks;
  std::vector<MemoryAccessInfoPtr> memoryAccesses;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : basicBlocks) it->accept(v);
  }
};

struct CallGraphInfo {
  std::vector<std::string> calls;
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;
  CallGraphInfoPtr callGraphInfo;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : functionInfos) it->accept(v);
  }
};
}  // namespace graph

class LLVMIRExtractor {
 public:
  LLVMIRExtractor(ClangDriverPtr clangDriver);

  graph::ExtractionInfoPtr GraphFromString(std::string src);
  seq::ExtractionInfoPtr SeqFromString(std::string src);

 private:
  ClangDriverPtr clangDriver_;
};

}  // namespace llvm
}  // namespace compy
