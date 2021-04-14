#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/ADT/StringRef.h"
#include <clang/Analysis/CFG.h>

#include "clang_extractor.h"

namespace compy {
namespace clang {
namespace graph {

/**
 * Keeps a queue of tokens that are not assigned to any AST-Graph node yet
 */
class TokenQueue {
 public:
  TokenQueue(::clang::Preprocessor &pp) : pp_(pp) {
    pp_.setTokenWatcher([this](auto token) { this->addToken(token); });
  }

  TokenQueue(TokenQueue const &) = delete;
  TokenQueue &operator=(TokenQueue const &) = delete;

  ~TokenQueue() { pp_.setTokenWatcher(nullptr); }

  std::vector<TokenInfo> popTokensForRange(::clang::SourceRange range);
  TokenInfo *getTokenAt(::clang::SourceLocation loc);

 private:
  void addToken(::clang::Token token);

  ::clang::Preprocessor &pp_;
  std::vector<TokenInfo> tokens_;
  std::uint64_t nextIndex = 0;
};

class ExtractorASTVisitor
    : public ::clang::RecursiveASTVisitor<ExtractorASTVisitor> {
 public:
  ExtractorASTVisitor(::clang::ASTContext &context,
                      ExtractionInfoPtr extractionInfo, TokenQueue &tokenQueue)
      : context_(context),
        extractionInfo_(extractionInfo),
        tokenQueue_(tokenQueue) {}

  bool VisitStmt(::clang::Stmt *s);
  bool VisitFunctionDecl(::clang::FunctionDecl *f);

  // postorder traversal is necessary so that tokens get assigned to
  // nodes closer to the leaves first
  bool shouldTraversePostOrder() const { return true; }

 private:
  FunctionInfoPtr getInfo(const ::clang::FunctionDecl &func);
  CFGBlockInfoPtr getInfo(const ::clang::CFGBlock &block);
  StmtInfoPtr getInfo(const ::clang::Stmt &stmt);
  DeclInfoPtr getInfo(const ::clang::Decl &decl, bool consumeTokens);

 private:
  ::clang::ASTContext &context_;
  ExtractionInfoPtr extractionInfo_;
  TokenQueue &tokenQueue_;

  std::unordered_map<const ::clang::Stmt *, StmtInfoPtr> stmtInfos_;
  std::unordered_map<const ::clang::CFGBlock *, CFGBlockInfoPtr> cfgBlockInfos_;
  std::unordered_map<const ::clang::Decl *, DeclInfoPtr> declInfos_;
};

class ExtractorASTConsumer : public ::clang::ASTConsumer {
 public:
  ExtractorASTConsumer(::clang::CompilerInstance &CI,
                       ExtractionInfoPtr extractionInfo);

  bool HandleTopLevelDecl(::clang::DeclGroupRef DR) override;

 private:
  ExtractorASTVisitor visitor_;
  TokenQueue tokenQueue_;
};

class ExtractorFrontendAction : public ::clang::ASTFrontendAction {
 public:
  std::unique_ptr<::clang::ASTConsumer> CreateASTConsumer(
      ::clang::CompilerInstance &CI, ::llvm::StringRef file) override;

  ExtractionInfoPtr extractionInfo;
};

}  // namespace graph
}  // namespace clang
}  // namespace compy
