#include "clang_seq_frontendaction.h"

#include <memory>
#include <string>
#include <vector>

#include "clang/AST/AST.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Rewrite/Core/TokenRewriter.h"
#include "llvm/ADT/StringRef.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace clang {
namespace seq {

void ExtractorASTVisitor::init() {
  mappedNames_.clear();
  num_functions_ = 0;
  num_variables_ = 0;
}

void ExtractorASTVisitor::setState(STATE state) { state_ = state; }

bool ExtractorASTVisitor::VisitFunctionDecl(FunctionDecl *f) {
  // Only proceed on function definitions, not declarations. Otherwise, all
  // function declarations in headers are traversed also.
  if (!f->hasBody() || !f->getDeclName().isIdentifier()) {
    return true;
  }

  if (state_ == STATE::Map) {
    mapName(*f);
  }

  else if (state_ == STATE::Capture) {
    FunctionInfoPtr functionInfo = getInfo(*f);
    extractionInfo_->functionInfos.push_back(functionInfo);

    // Get string of function.
    SourceRange sourceRange = f->getSourceRange();

    CharSourceRange charSourceRange = ::clang::Lexer::getAsCharRange(
        sourceRange, context_.getSourceManager(), context_.getLangOpts());

    std::string str = ::clang::Lexer::getSourceText(charSourceRange,
                                                    context_.getSourceManager(),
                                                    context_.getLangOpts())
                          .str();

    // Create file id in source manager. So it will follow the same lex flow as
    // regular files.
    FileID fid = context_.getSourceManager().createFileID(
        MemoryBuffer::getMemBuffer(str));

    // Create lexer.
    ::clang::Lexer lex(context_.getSourceManager().getLocForStartOfFile(fid),
                       context_.getLangOpts(), str.data(), str.data(),
                       str.data() + str.size());

    // Lex function.
    Token tok;
    lex.LexFromRawLexer(tok);
    while (true) {
      // Add to tokens.
      TokenInfoPtr tokenInfo(new TokenInfo());
      functionInfo->tokenInfos.push_back(tokenInfo);

      // Get string token.
      // - Get token from lexer.
      std::string strToken = ::clang::Lexer::getSpelling(
          tok, context_.getSourceManager(), context_.getLangOpts(), nullptr);

      // - Check if there exists a mapping for this token.
      auto it = mappedNames_.find(strToken);
      if (it != mappedNames_.end()) strToken = it->second;

      tokenInfo->name = strToken;

      // Get token kind.
      tokenInfo->kind = tok.getName();

      // Check if done and get next token if not.
      if (tok.getLocation() == sourceRange.getEnd() || tok.is(tok::eof)) {
        break;
      }
      lex.LexFromRawLexer(tok);
    }
  }

  return RecursiveASTVisitor<ExtractorASTVisitor>::VisitFunctionDecl(f);
}

bool ExtractorASTVisitor::VisitVarDecl(VarDecl *decl) {
  if (state_ == STATE::Map) {
    mapName(*decl);
  }

  return RecursiveASTVisitor<ExtractorASTVisitor>::VisitVarDecl(decl);
}

FunctionInfoPtr ExtractorASTVisitor::getInfo(const FunctionDecl &func) {
  FunctionInfoPtr info(new FunctionInfo());

  // Collect name.
  info->name = func.getNameAsString();

  return info;
}

std::string ExtractorASTVisitor::mapName(const NamedDecl &decl) {
  std::string name = decl.getNameAsString();

  auto it = mappedNames_.find(name);
  if (it != mappedNames_.end()) return it->second;

  std::string mappedName;
  if (isa<FunctionDecl>(decl)) {
    mappedName = "fn_" + std::to_string(num_functions_);
    num_functions_++;
  } else if (isa<VarDecl>(decl)) {
    mappedName = "var_" + std::to_string(num_variables_);
    num_variables_++;
  }
  mappedNames_[name] = mappedName;

  return mappedName;
}

ExtractorASTConsumer::ExtractorASTConsumer(ASTContext &context,
                                           ExtractionInfoPtr extractionInfo)
    : visitor_(context, extractionInfo) {}

bool ExtractorASTConsumer::HandleTopLevelDecl(DeclGroupRef DR) {
  for (auto it = DR.begin(), e = DR.end(); it != e; ++it) {
    visitor_.setState(ExtractorASTVisitor::STATE::Map);
    visitor_.TraverseDecl(*it);

    visitor_.setState(ExtractorASTVisitor::STATE::Capture);
    visitor_.TraverseDecl(*it);
  }

  return true;
}

std::unique_ptr<ASTConsumer> ExtractorFrontendAction::CreateASTConsumer(
    CompilerInstance &CI, StringRef file) {
  extractionInfo.reset(new ExtractionInfo());

  return std::make_unique<ExtractorASTConsumer>(CI.getASTContext(),
                                                extractionInfo);
}

}  // namespace seq
}  // namespace clang
}  // namespace compy
