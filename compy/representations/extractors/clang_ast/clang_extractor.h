#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common/clang_driver.h"
#include "common/visitor.h"

namespace compy {
namespace clang {

namespace seq {
struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct TokenInfo;
using TokenInfoPtr = std::shared_ptr<TokenInfo>;

struct TokenInfo : IVisitee {
  std::string name;
  std::string kind;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::vector<TokenInfoPtr> tokenInfos;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : tokenInfos) it->accept(v);
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

struct DeclInfo;
using DeclInfoPtr = std::shared_ptr<DeclInfo>;

struct StmtInfo;
using StmtInfoPtr = std::shared_ptr<StmtInfo>;

struct CFGBlockInfo;
using CFGBlockInfoPtr = std::shared_ptr<CFGBlockInfo>;

struct FunctionInfo;
using FunctionInfoPtr = std::shared_ptr<FunctionInfo>;

struct ExtractionInfo;
using ExtractionInfoPtr = std::shared_ptr<ExtractionInfo>;

struct TokenInfo : IVisitee {
  std::uint64_t index;
  std::string name;
  std::string kind;
  ::clang::SourceLocation location;

  void accept(IVisitor* v) override { v->visit(this); }
};

struct OperandInfo : IVisitee {
  virtual ~OperandInfo() = default;
};

struct DeclInfo : OperandInfo {
  std::string name;
  std::string type;
  std::string kind;
  std::vector<TokenInfo> tokens;
  TokenInfo nameToken;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (auto& it : tokens) it.accept(v);
  }
};

struct StmtInfo : OperandInfo {
  std::string name;
  std::vector<TokenInfo> tokens;
  std::string operation;
  std::vector<OperandInfoPtr> ast_relations;
  std::vector<OperandInfoPtr> ref_relations;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (auto& it : tokens) it.accept(v);
    for (const auto& it : ast_relations) it->accept(v);
  }
};

struct CFGBlockInfo {
  std::string name;
  std::vector<StmtInfoPtr> statements;
  std::vector<CFGBlockInfoPtr> successors;
};

struct FunctionInfo : IVisitee {
  std::string name;
  std::string type;
  std::vector<TokenInfo> tokens;
  std::vector<DeclInfoPtr> args;
  std::vector<CFGBlockInfoPtr> cfgBlocks;
  StmtInfoPtr entryStmt;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (auto& it : tokens) it.accept(v);
    for (const auto& it : args) it->accept(v);
    entryStmt->accept(v);
  }
};

struct ExtractionInfo : IVisitee {
  std::vector<FunctionInfoPtr> functionInfos;

  void accept(IVisitor* v) override {
    v->visit(this);
    for (const auto& it : functionInfos) it->accept(v);
  }
};
}  // namespace graph

class ClangExtractor {
 public:
  ClangExtractor(ClangDriverPtr clangDriver);

  graph::ExtractionInfoPtr GraphFromString(std::string src);
  seq::ExtractionInfoPtr SeqFromString(std::string src);

 private:
  ClangDriverPtr clangDriver_;
};

}  // namespace clang
}  // namespace compy
