#pragma once

#include <memory>
#include <string>
#include <vector>

#include <clang/Analysis/CFG.h>
#include "clang/AST/AST.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang_extractor.h"
#include "llvm/ADT/StringRef.h"

namespace compy {
namespace clang {
namespace graph {

class ExtractorASTVisitor
    : public ::clang::RecursiveASTVisitor<ExtractorASTVisitor> {
 public:
  ExtractorASTVisitor(::clang::ASTContext &context,
                      ExtractionInfoPtr extractionInfo)
      : context_(context), extractionInfo_(extractionInfo) {}

  bool VisitStmt(::clang::Stmt *s);
  bool VisitFunctionDecl(::clang::FunctionDecl *f);

 private:
  FunctionInfoPtr getInfo(const ::clang::FunctionDecl &func);
  CFGBlockInfoPtr getInfo(const ::clang::CFGBlock &block);
  StmtInfoPtr getInfo(const ::clang::Stmt &stmt);
  DeclInfoPtr getInfo(const ::clang::Decl &decl);

 private:
  ::clang::ASTContext &context_;
  ExtractionInfoPtr extractionInfo_;

  std::unordered_map<const ::clang::Stmt *, StmtInfoPtr> stmtInfos_;
  std::unordered_map<const ::clang::CFGBlock *, CFGBlockInfoPtr> cfgBlockInfos_;
  std::unordered_map<const ::clang::Decl *, DeclInfoPtr> declInfos_;
};

class ExtractorASTConsumer : public ::clang::ASTConsumer {
 public:
  ExtractorASTConsumer(::clang::ASTContext &context,
                       ExtractionInfoPtr extractionInfo);

  bool HandleTopLevelDecl(::clang::DeclGroupRef DR) override;

 private:
  ExtractorASTVisitor visitor_;
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
