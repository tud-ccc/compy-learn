#include "llvm_graph_funcinfo.h"

#include <iostream>
#include <sstream>

#include "llvm/IR/Instructions.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace graph {

std::string llvmTypeToString(Type *type) {
  std::string typeName;
  raw_string_ostream rso(typeName);
  type->print(rso);
  return rso.str();
}

/**
 * Get a unique Name for an LLVM value.
 *
 * This function should always be used instead of the values getName()
 * function. If the object has no name yet, a new unique name is generated
 * based on the default name.
 */
std::string FunctionInfoPass::getUniqueName(const Value &v) {
  if (v.hasName()) return v.getName();

  auto iter = valueNames.find(&v);
  if (iter != valueNames.end()) return iter->second;

  std::stringstream ss;
  if (isa<Value>(v))
    ss << "val";
  else if (isa<BasicBlock>(v))
    ss << "bb";
  else if (isa<Function>(v))
    ss << "func";
  else
    ss << "v";

  ss << valueNames.size();

  valueNames[&v] = ss.str();
  return ss.str();
}

ArgInfoPtr FunctionInfoPass::getInfo(const Argument &arg) {
  auto it = argInfos.find(&arg);
  if (it != argInfos.end()) return it->second;

  ArgInfoPtr info(new ArgInfo());
  argInfos[&arg] = info;

  info->name = getUniqueName(arg);

  // collect the type
  info->type = llvmTypeToString(arg.getType());

  return info;
}

ConstantInfoPtr FunctionInfoPass::getInfo(const ::llvm::Constant &con) {
  auto it = constantInfos.find(&con);
  if (it != constantInfos.end()) return it->second;

  ConstantInfoPtr info(new ConstantInfo());
  constantInfos[&con] = info;

  // collect the type
  info->type = llvmTypeToString(con.getType());

  return info;
}

InstructionInfoPtr FunctionInfoPass::getInfo(const Instruction &inst) {
  auto it = instructionInfos.find(&inst);
  if (it != instructionInfos.end()) return it->second;

  InstructionInfoPtr info(new InstructionInfo());
  instructionInfos[&inst] = info;

  // collect opcode
  info->opcode = inst.getOpcodeName();

  if (inst.getOpcodeName() == std::string("ret")) {
    info_->exitInstructions.push_back(info);
  }

  // collect type
  std::string typeName;
  raw_string_ostream rso(typeName);
  inst.getType()->print(rso);
  info->type = rso.str();

  // collect data dependencies
  for (auto &use : inst.operands()) {
    if (isa<Instruction>(use.get())) {
      auto &opInst = *cast<Instruction>(use.get());
      info->operands.push_back(getInfo(opInst));
    }

    if (isa<Argument>(use.get())) {
      auto &opInst = *cast<Argument>(use.get());
      info->operands.push_back(getInfo(opInst));
    }

    if (isa<Constant>(use.get())) {
      auto &opInst = *cast<Constant>(use.get());
      info->operands.push_back(getInfo(opInst));
    }
  }

  // collect called function (if this instruction is a call)
  if (isa<CallInst>(inst)) {
    auto &call = cast<CallInst>(inst);
    Function *calledFunction = call.getCalledFunction();
    if (calledFunction != nullptr) {
      info->callTarget = getUniqueName(*calledFunction);
    }
  }

  // load or store?
  info->isLoadOrStore = false;
  if (isa<LoadInst>(inst)) info->isLoadOrStore = true;
  if (isa<StoreInst>(inst)) info->isLoadOrStore = true;

  // collect function this instruction belongs to
  info->function = info_;

  return info;
}

BasicBlockInfoPtr FunctionInfoPass::getInfo(const BasicBlock &bb) {
  auto it = basicBlockInfos.find(&bb);
  if (it != basicBlockInfos.end()) return it->second;

  BasicBlockInfoPtr info(new BasicBlockInfo());
  basicBlockInfos[&bb] = info;

  info->name = getUniqueName(bb);

  // collect all successors
  auto term = bb.getTerminator();
  for (size_t i = 0; i < term->getNumSuccessors(); i++) {
    BasicBlock *succ = term->getSuccessor(i);
    info->successors.push_back(getInfo(*succ));
  }

  return info;
}

MemoryAccessInfoPtr FunctionInfoPass::getInfo(MemoryAccess &acc) {
  auto it = memoryAccessInfos.find(&acc);
  if (it != memoryAccessInfos.end()) return it->second;

  MemoryAccessInfoPtr info(new MemoryAccessInfo());
  memoryAccessInfos[&acc] = info;

  info->block = getInfo(*acc.getBlock());

  if (isa<MemoryUseOrDef>(acc)) {
    if (isa<MemoryUse>(acc))
      info->type = "use";
    else
      info->type = "def";

    auto inst = cast<MemoryUseOrDef>(acc).getMemoryInst();
    if (inst != nullptr) {
      info->inst = getInfo(*inst);
    } else {
      info->inst = NULL;
      assert(info->type == "def");
      info->type = "live on entry";
    }

    auto dep = cast<MemoryUseOrDef>(acc).getDefiningAccess();
    if (dep != nullptr) {
      info->dependencies.push_back(getInfo(*dep));
    }
  } else {
    info->type = "phi";
    info->inst = NULL;
    auto &phi = cast<MemoryPhi>(acc);
    for (unsigned i = 0; i < phi.getNumIncomingValues(); i++) {
      auto dep = phi.getIncomingValue(i);
      info->dependencies.push_back(getInfo(*dep));
    }
  }

  return info;
}

bool FunctionInfoPass::runOnFunction(::llvm::Function &func) {
  // wipe all data from the previous run
  valueNames.clear();
  argInfos.clear();
  basicBlockInfos.clear();
  instructionInfos.clear();
  memoryAccessInfos.clear();
  valueNames.clear();

  // create a new info object and invalidate the old one
  info_ = FunctionInfoPtr(new FunctionInfo());

  info_->name = getUniqueName(func);
  info_->entryInstruction =
      getInfo(*func.getEntryBlock().getInstList().begin());

  std::string rtypeName;
  raw_string_ostream rso(rtypeName);
  func.getReturnType()->print(rso);
  info_->type = rso.str();

  // collect all basic blocks and their instructions
  for (auto &bb : func.getBasicBlockList()) {
    BasicBlockInfoPtr bbInfo = getInfo(bb);
    for (auto &inst : bb) {
      bbInfo->instructions.push_back(getInfo(inst));
    }
    info_->basicBlocks.push_back(bbInfo);
  }

  // collect all arguments
  for (auto &arg : func.args()) {
    info_->args.push_back(getInfo(arg));
  }

  // dump app memory accesses
  auto &mssaPass = getAnalysis<MemorySSAWrapperPass>();
  auto &mssa = mssaPass.getMSSA();
  for (auto &bb : func.getBasicBlockList()) {
    // live on entry
    auto entry = mssa.getLiveOnEntryDef();
    info_->memoryAccesses.push_back(getInfo(*entry));

    // memory phis
    auto phi = mssa.getMemoryAccess(&bb);
    if (phi != nullptr) {
      info_->memoryAccesses.push_back(getInfo(*phi));
    }

    // memory use or defs
    for (auto &inst : bb) {
      auto access = mssa.getMemoryAccess(&inst);
      if (access != nullptr) {
        info_->memoryAccesses.push_back(getInfo(*access));
      }
    }
  }

  // indicate that nothing was changed
  return false;
}

void FunctionInfoPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<MemorySSAWrapperPass>();
  au.setPreservesAll();
}

char FunctionInfoPass::ID = 0;

static RegisterPass<FunctionInfoPass> X("funcinfo", "Function Info Extractor",
                                        true /* Only looks at CFG */,
                                        true /* Analysis Pass */);

}  // namespace graph
}  // namespace llvm
}  // namespace compy
