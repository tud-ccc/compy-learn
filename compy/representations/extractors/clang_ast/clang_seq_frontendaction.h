#pragma once

#include <memory>
#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/ADT/StringRef.h"

#include "clang_extractor.h"

namespace compy {
namespace clang {
namespace seq {

class ExtractorASTVisitor
    : public ::clang::RecursiveASTVisitor<ExtractorASTVisitor> {
 public:
  enum STATE {
    Map = 0,
    Capture = 1,
  };

 public:
  ExtractorASTVisitor(::clang::ASTContext &context,
                      ExtractionInfoPtr extractionInfo)
      : state_(STATE::Map), context_(context), extractionInfo_(extractionInfo) {
    init();
  }

  void init();
  void setState(STATE state);

  bool VisitFunctionDecl(::clang::FunctionDecl *f);
  bool VisitVarDecl(::clang::VarDecl *decl);

 private:
  FunctionInfoPtr getInfo(const ::clang::FunctionDecl &func);

  std::string mapName(const ::clang::NamedDecl &decl);

 private:
  STATE state_;
  ::clang::ASTContext &context_;
  ExtractionInfoPtr extractionInfo_;

  std::unordered_map<std::string, std::string> mappedNames_;
  unsigned int num_functions_;
  unsigned int num_variables_;
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

}  // namespace seq
}  // namespace clang
}  // namespace compy
