#include "clang_graph_frontendaction.h"

#include <iostream>
#include <utility>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "clang/Analysis/CFG.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {
namespace clang {
namespace graph {

bool ExtractorASTVisitor::VisitStmt(Stmt *s) {
  StmtInfoPtr info = getInfo(*s);

  // Add child stmts
  for (auto it : s->children()) {
    if (it) {
      StmtInfoPtr childInfo = getInfo(*it);
      info->ast_relations.push_back(childInfo);
    }
  }

  return RecursiveASTVisitor<ExtractorASTVisitor>::VisitStmt(s);
}

bool ExtractorASTVisitor::VisitFunctionDecl(FunctionDecl *f) {
  // Only proceed on function definitions, not declarations. Otherwise, all
  // function declarations in headers are traversed also.
  if (!f->hasBody() || !f->getDeclName().isIdentifier()) {
    return true;
  }

  //  ::llvm::errs() << f->getNameAsString() << "\n";

  FunctionInfoPtr functionInfo = getInfo(*f);
  extractionInfo_->functionInfos.push_back(functionInfo);

  // Add entry stmt.
  functionInfo->entryStmt = getInfo(*f->getBody());

  // Add args.
  for (auto it : f->parameters()) {
    functionInfo->args.push_back(getInfo(*it));
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
    info->ref_relations.push_back(getInfo(*de->getDecl()));
  }

  return info;
}

DeclInfoPtr ExtractorASTVisitor::getInfo(const Decl &decl) {
  auto it = declInfos_.find(&decl);
  if (it != declInfos_.end()) return it->second;

  DeclInfoPtr info(new DeclInfo);
  declInfos_[&decl] = info;

  // Collect name.
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(&decl)) {
    info->name = vd->getQualifiedNameAsString();
  }

  // Collect type.
  if (const ValueDecl *vd = dyn_cast<ValueDecl>(&decl)) {
    info->type = vd->getType().getAsString();
  }

  return info;
}

ExtractorASTConsumer::ExtractorASTConsumer(ASTContext &context,
                                           ExtractionInfoPtr extractionInfo)
    : visitor_(context, extractionInfo) {}

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

  return std::make_unique<ExtractorASTConsumer>(CI.getASTContext(),
                                                extractionInfo);
}

}  // namespace graph
}  // namespace clang
}  // namespace compy
