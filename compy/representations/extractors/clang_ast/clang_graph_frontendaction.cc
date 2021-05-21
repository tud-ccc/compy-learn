#include "clang_graph_frontendaction.h"

#include <exception>
#include <iostream>
#include <utility>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/CFG.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace clang {
namespace graph {

bool ExtractorASTVisitor::VisitStmt(Stmt *s) {
  // Collect child stmts
  std::vector<OperandInfoPtr> ast_relations;
  for (auto it : s->children()) {
    if (it) {
      StmtInfoPtr childInfo = getInfo(*it);
      ast_relations.push_back(childInfo);
    }
  }

  if (auto *ds = dyn_cast<DeclStmt>(s)) {
    for (const Decl *decl : ds->decls()) {
      ast_relations.push_back(getInfo(*decl, false));
    }
  }

  StmtInfoPtr info = getInfo(*s);
  info->ast_relations.insert(info->ast_relations.end(), ast_relations.begin(),
                             ast_relations.end());

  return RecursiveASTVisitor<ExtractorASTVisitor>::VisitStmt(s);
}

bool ExtractorASTVisitor::VisitFunctionDecl(FunctionDecl *f) {
  // Only proceed on function definitions, not declarations. Otherwise, all
  // function declarations in headers are traversed also.
  if (!f->hasBody() || !f->getDeclName().isIdentifier()) {
    // throw away the tokens
    tokenQueue_.popTokensForRange(f->getSourceRange());
    return true;
  }

  //  ::llvm::errs() << f->getNameAsString() << "\n";

  FunctionInfoPtr functionInfo = getInfo(*f);
  extractionInfo_->functionInfos.push_back(functionInfo);

  // Add entry stmt.
  functionInfo->entryStmt = getInfo(*f->getBody());

  // Add args.
  for (auto it : f->parameters()) {
    functionInfo->args.push_back(getInfo(*it, true));
  }

  // Dump CFG.
  std::unique_ptr<CFG> cfg =
      CFG::buildCFG(f, f->getBody(), &context_, CFG::BuildOptions());
  //  cfg->dump(LangOptions(), true);

  // Create CFG Blocks.
  for (CFG::iterator it = cfg->begin(), Eb = cfg->end(); it != Eb; ++it) {
    CFGBlock *B = *it;
    functionInfo->cfgBlocks.push_back(getInfo(*B));
  }

  return RecursiveASTVisitor<ExtractorASTVisitor>::VisitFunctionDecl(f);
}

CFGBlockInfoPtr ExtractorASTVisitor::getInfo(const ::clang::CFGBlock &block) {
  auto it = cfgBlockInfos_.find(&block);
  if (it != cfgBlockInfos_.end()) return it->second;

  CFGBlockInfoPtr info(new CFGBlockInfo);
  cfgBlockInfos_[&block] = info;

  // Collect name.
  info->name = "cfg_" + std::to_string(block.getBlockID());

  // Collect statements.
  for (CFGBlock::const_iterator it = block.begin(), Es = block.end(); it != Es;
       ++it) {
    if (Optional<CFGStmt> CS = it->getAs<CFGStmt>()) {
      const Stmt *S = CS->getStmt();
      info->statements.push_back(getInfo(*S));
    }
  }
  if (block.getTerminatorStmt()) {
    const Stmt *S = block.getTerminatorStmt();
    info->statements.push_back(getInfo(*S));
  }

  // Collect successors.
  for (CFGBlock::const_succ_iterator it = block.succ_begin(),
                                     Es = block.succ_end();
       it != Es; ++it) {
    CFGBlock *B = *it;
    if (B) info->successors.push_back(getInfo(*B));
  }

  return info;
}

FunctionInfoPtr ExtractorASTVisitor::getInfo(const FunctionDecl &func) {
  FunctionInfoPtr info(new FunctionInfo());

  // Collect name.
  info->name = func.getNameAsString();

  // Collect type.
  info->type = func.getType().getAsString();

  // Collect tokens
  info->tokens = tokenQueue_.popTokensForRange(func.getSourceRange());

  return info;
}

StmtInfoPtr ExtractorASTVisitor::getInfo(const Stmt &stmt) {
  auto it = stmtInfos_.find(&stmt);
  if (it != stmtInfos_.end()) return it->second;

  StmtInfoPtr info(new StmtInfo);
  stmtInfos_[&stmt] = info;

  // Collect name.
  info->name = stmt.getStmtClassName();

  // Collect referencing targets.
  if (const DeclRefExpr *de = dyn_cast<DeclRefExpr>(&stmt)) {
    info->ref_relations.push_back(getInfo(*de->getDecl(), false));
  }

  // Collect tokens
  info->tokens = tokenQueue_.popTokensForRange(stmt.getSourceRange());

  return info;
}

DeclInfoPtr ExtractorASTVisitor::getInfo(const Decl &decl, bool consumeTokens) {
  auto it = declInfos_.find(&decl);
  if (it != declInfos_.end()) {
    if (consumeTokens) {
      auto tokens = tokenQueue_.popTokensForRange(decl.getSourceRange());
      it->second->tokens.insert(it->second->tokens.end(), tokens.begin(),
                                tokens.end());
    }

    return it->second;
  }

  DeclInfoPtr info(new DeclInfo);
  declInfos_[&decl] = info;

  info->kind = decl.getDeclKindName();

  // Collect name.
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(&decl)) {
    info->name = vd->getQualifiedNameAsString();

    if (const auto nameTokenPtr = tokenQueue_.getTokenAt(vd->getLocation())) {
      info->nameToken = *nameTokenPtr;
    }
  }

  // Collect type.
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(&decl)) {
    info->type = vd->getType().getAsString();
  }

  // Collect tokens
  if (consumeTokens) {
    info->tokens = tokenQueue_.popTokensForRange(decl.getSourceRange());
  }

  return info;
}

ExtractorASTConsumer::ExtractorASTConsumer(CompilerInstance &CI,
                                           ExtractionInfoPtr extractionInfo)
    : visitor_(CI.getASTContext(), std::move(extractionInfo), tokenQueue_),
      tokenQueue_(CI.getPreprocessor()) {}

bool ExtractorASTConsumer::HandleTopLevelDecl(DeclGroupRef DR) {
  for (auto it = DR.begin(), e = DR.end(); it != e; ++it) {
    visitor_.TraverseDecl(*it);
  }

  return true;
}

std::unique_ptr<ASTConsumer> ExtractorFrontendAction::CreateASTConsumer(
    CompilerInstance &CI, StringRef file) {
  extractionInfo.reset(new ExtractionInfo());
  //  CI.getASTContext().getLangOpts().OpenCL
  return std::make_unique<ExtractorASTConsumer>(CI, extractionInfo);
}

std::vector<TokenInfo> TokenQueue::popTokensForRange(
    ::clang::SourceRange range) {
  std::vector<TokenInfo> result;
  auto startPos = token_index_[range.getBegin().getRawEncoding()];
  auto endPos = token_index_[range.getEnd().getRawEncoding()];
  for (std::size_t i = startPos; i <= endPos; ++i) {
    if (token_consumed_[i]) continue;

    result.push_back(tokens_[i]);
    token_consumed_[i] = true;
  }

  return result;
}

void TokenQueue::addToken(::clang::Token token) {
  TokenInfo info;
  info.index = nextIndex++;
  info.kind = token.getName();
  info.name = pp_.getSpelling(token, nullptr);
  info.location = token.getLocation();
  tokens_.push_back(info);
  token_consumed_.push_back(false);
  token_index_[info.location.getRawEncoding()] = tokens_.size() - 1;
}

TokenInfo *TokenQueue::getTokenAt(SourceLocation loc) {
  auto pos = token_index_.find(loc.getRawEncoding());
  if (pos == token_index_.end()) return nullptr;

  return &tokens_[pos->second];
}

}  // namespace graph
}  // namespace clang
}  // namespace compy
