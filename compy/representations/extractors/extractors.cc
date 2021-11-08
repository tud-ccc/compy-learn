#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clang_ast/clang_extractor.h"
#include "common/clang_driver.h"
#include "llvm_ir/llvm_extractor.h"

using namespace compy;

namespace py = pybind11;
namespace cg = compy::clang::graph;
namespace cs = compy::clang::seq;
namespace lg = compy::llvm::graph;
namespace ls = compy::llvm::seq;

using CD = compy::ClangDriver;
using CE = compy::clang::ClangExtractor;
using LE = compy::llvm::LLVMIRExtractor;

namespace pybind11 {
template <>
struct polymorphic_type_hook<lg::OperandInfo> {
  static const void *get(const lg::OperandInfo *src,
                         const std::type_info *&type) {
    if (src) {
      if (dynamic_cast<const lg::ArgInfo *>(src) != nullptr) {
        type = &typeid(lg::ArgInfo);
        return static_cast<const lg::ArgInfo *>(src);
      }
      if (dynamic_cast<const lg::InstructionInfo *>(src) != nullptr) {
        type = &typeid(lg::InstructionInfo);
        return static_cast<const lg::InstructionInfo *>(src);
      }
      if (dynamic_cast<const lg::ConstantInfo *>(src) != nullptr) {
        type = &typeid(lg::ConstantInfo);
        return static_cast<const lg::ConstantInfo *>(src);
      }
    }
    return src;
  }
};

template <>
struct polymorphic_type_hook<cg::OperandInfo> {
  static const void *get(const cg::OperandInfo *src,
                         const std::type_info *&type) {
    if (src) {
      if (dynamic_cast<const cg::DeclInfo *>(src) != nullptr) {
        type = &typeid(cg::DeclInfo);
        return static_cast<const cg::DeclInfo *>(src);
      }
      if (dynamic_cast<const cg::StmtInfo *>(src) != nullptr) {
        type = &typeid(cg::StmtInfo);
        return static_cast<const cg::StmtInfo *>(src);
      }
    }
    return src;
  }
};
}  // namespace pybind11

class PyVisitor : public IVisitor {
 public:
  using IVisitor::IVisitor; /* Inherit the constructors */

  void visit(IVisitee *v) override {
    PYBIND11_OVERLOAD_PURE(
        void,     /* Return type */
        IVisitor, /* Parent class */
        visit,    /* Name of function in C++ (must match Python name) */
        v         /* Argument(s) */
    );
  }
};

void registerClangDriver(py::module m) {
  py::class_<CD, std::shared_ptr<CD>> clangDriver(m, "ClangDriver");
  clangDriver
      .def(py::init<CD::ProgrammingLanguage, CD::OptimizationLevel,
                    std::vector<std::tuple<std::string, CD::IncludeDirType>>,
                    std::vector<std::string>>())
      .def("addIncludeDir", &CD::addIncludeDir)
      .def("removeIncludeDir", &CD::removeIncludeDir)
      .def("getFileName", &CD::getFileName)
      .def("setFileName", &CD::setFileName)
      .def("getCompilerBinary", &CD::getCompilerBinary)
      .def("setCompilerBinary", &CD::setCompilerBinary);

  py::enum_<CD::ProgrammingLanguage>(clangDriver, "ProgrammingLanguage")
      .value("C", CD::ProgrammingLanguage::C)
      .value("CPlusPlus", CD::ProgrammingLanguage::CPLUSPLUS)
      .value("OpenCL", CD::ProgrammingLanguage::OPENCL)
      .value("LLVM", CD::ProgrammingLanguage::LLVM)
      .export_values();

  py::enum_<CD::OptimizationLevel>(clangDriver, "OptimizationLevel")
      .value("O0", CD::OptimizationLevel::O0)
      .value("O1", CD::OptimizationLevel::O1)
      .value("O2", CD::OptimizationLevel::O2)
      .value("O3", CD::OptimizationLevel::O3)
      .export_values();

  py::enum_<CD::IncludeDirType>(clangDriver, "IncludeDirType")
      .value("System", CD::IncludeDirType::SYSTEM)
      .value("User", CD::IncludeDirType::USER)
      .export_values();
}

void registerClangExtractor(py::module m_parent) {
  // Extractor
  py::class_<CE> clangExtractor(m_parent, "ClangExtractor");
  clangExtractor.def(py::init<ClangDriverPtr>());
  clangExtractor.def("GraphFromString", &CE::GraphFromString);
  clangExtractor.def("SeqFromString", &CE::SeqFromString);

  py::module m = m_parent.def_submodule("clang");

  // Subtypes
  py::module m_graph = m.def_submodule("graph");

  // Graph extractor
  py::class_<cg::ExtractionInfo, std::shared_ptr<cg::ExtractionInfo>>(
      m_graph, "ExtractionInfo")
      .def("accept", &cg::ExtractionInfo::accept)
      .def_readonly("functionInfos", &cg::ExtractionInfo::functionInfos);

  py::class_<cg::DeclInfo, std::shared_ptr<cg::DeclInfo>>(m_graph, "DeclInfo")
      .def_readonly("name", &cg::DeclInfo::name)
      .def_readonly("kind", &cg::DeclInfo::kind)
      .def_readonly("nameToken", &cg::DeclInfo::nameToken)
      .def_readonly("tokens", &cg::DeclInfo::tokens)
      .def_readonly("type", &cg::DeclInfo::type);

  py::class_<cg::FunctionInfo, std::shared_ptr<cg::FunctionInfo>>(
      m_graph, "FunctionInfo")
      .def("accept", &cg::FunctionInfo::accept)
      .def_readonly("name", &cg::FunctionInfo::name)
      .def_readonly("tokens", &cg::FunctionInfo::tokens)
      .def_readonly("type", &cg::FunctionInfo::type)
      .def_readonly("args", &cg::FunctionInfo::args)
      .def_readonly("cfgBlocks", &cg::FunctionInfo::cfgBlocks)
      .def_readonly("entryStmt", &cg::FunctionInfo::entryStmt);

  py::class_<cg::CFGBlockInfo, std::shared_ptr<cg::CFGBlockInfo>>(
      m_graph, "CFGBlockInfo")
      .def_readonly("name", &cg::CFGBlockInfo::name)
      .def_readonly("statements", &cg::CFGBlockInfo::statements)
      .def_readonly("successors", &cg::CFGBlockInfo::successors);

  py::class_<cg::StmtInfo, std::shared_ptr<cg::StmtInfo>>(m_graph, "StmtInfo")
      .def_readonly("name", &cg::StmtInfo::name)
      .def_readonly("tokens", &cg::StmtInfo::tokens)
      .def_readonly("ast_relations", &cg::StmtInfo::ast_relations)
      .def_readonly("ref_relations", &cg::StmtInfo::ref_relations);

  py::class_<cg::TokenInfo, std::shared_ptr<cg::TokenInfo>>(m_graph,
                                                            "TokenInfo")
      .def_readonly("name", &cg::TokenInfo::name)
      .def_readonly("kind", &cg::TokenInfo::kind)
      .def_readonly("index", &cg::TokenInfo::index);

  // Sequence extractor
  py::module m_seq = m.def_submodule("seq");

  py::class_<cs::ExtractionInfo, std::shared_ptr<cs::ExtractionInfo>>(
      m_seq, "ExtractionInfo")
      .def("accept", &cs::ExtractionInfo::accept)
      .def_readonly("functionInfos", &cs::ExtractionInfo::functionInfos);

  py::class_<cs::FunctionInfo, std::shared_ptr<cs::FunctionInfo>>(
      m_seq, "FunctionInfo")
      .def("accept", &cs::FunctionInfo::accept)
      .def_readonly("name", &cs::FunctionInfo::name)
      .def_readonly("tokenInfos", &cs::FunctionInfo::tokenInfos);

  py::class_<cs::TokenInfo, std::shared_ptr<cs::TokenInfo>>(m_seq, "TokenInfo")
      .def_readonly("name", &cs::TokenInfo::name)
      .def_readonly("kind", &cs::TokenInfo::kind);
}

void registerLLVMExtractor(py::module m_parent) {
  // Extractor
  py::class_<LE> llvmExtractor(m_parent, "LLVMIRExtractor");
  llvmExtractor.def(py::init<ClangDriverPtr>());
  llvmExtractor.def("GraphFromString", &LE::GraphFromString);
  llvmExtractor.def("SeqFromString", &LE::SeqFromString);

  // Subtypes
  py::module m = m_parent.def_submodule("llvm");

  // Graph extractor
  py::module m_graph = m.def_submodule("graph");

  py::class_<lg::ExtractionInfo, std::shared_ptr<lg::ExtractionInfo>>(
      m_graph, "ExtractionInfo")
      .def("accept", &lg::ExtractionInfo::accept)
      .def_readonly("functionInfos", &lg::ExtractionInfo::functionInfos)
      .def_readonly("callGraphInfo", &lg::ExtractionInfo::callGraphInfo);

  py::class_<lg::InstructionInfo, std::shared_ptr<lg::InstructionInfo>>(
      m_graph, "InstructionInfo")
      .def_readonly("type", &lg::InstructionInfo::type)
      .def_readonly("opcode", &lg::InstructionInfo::opcode)
      .def_readonly("callTarget", &lg::InstructionInfo::callTarget)
      .def_readonly("isLoadOrStore", &lg::InstructionInfo::isLoadOrStore)
      .def_readonly("operands", &lg::InstructionInfo::operands)
      .def_readonly("function", &lg::InstructionInfo::function);

  py::class_<lg::MemoryAccessInfo, std::shared_ptr<lg::MemoryAccessInfo>>(
      m_graph, "MemoryAccessInfo")
      .def_readonly("type", &lg::MemoryAccessInfo::type)
      .def_readonly("inst", &lg::MemoryAccessInfo::inst)
      .def_readonly("block", &lg::MemoryAccessInfo::block)
      .def_readonly("dependencies", &lg::MemoryAccessInfo::dependencies);

  py::class_<lg::BasicBlockInfo, std::shared_ptr<lg::BasicBlockInfo>>(
      m_graph, "BasicBlockInfo")
      .def_readonly("name", &lg::BasicBlockInfo::name)
      .def_readonly("instructions", &lg::BasicBlockInfo::instructions)
      .def_readonly("successors", &lg::BasicBlockInfo::successors);

  py::class_<lg::FunctionInfo, std::shared_ptr<lg::FunctionInfo>>(
      m_graph, "FunctionInfo")
      .def("accept", &lg::FunctionInfo::accept)
      .def_readonly("name", &lg::FunctionInfo::name)
      .def_readonly("type", &lg::FunctionInfo::type)
      .def_readonly("entryInstruction", &lg::FunctionInfo::entryInstruction)
      .def_readonly("exitInstructions", &lg::FunctionInfo::exitInstructions)
      .def_readonly("args", &lg::FunctionInfo::args)
      .def_readonly("basicBlocks", &lg::FunctionInfo::basicBlocks)
      .def_readonly("memoryAccesses", &lg::FunctionInfo::memoryAccesses);

  py::class_<lg::CallGraphInfo, std::shared_ptr<lg::CallGraphInfo>>(
      m_graph, "CallGraphInfo")
      .def_readonly("calls", &lg::CallGraphInfo::calls);

  py::class_<lg::ArgInfo, std::shared_ptr<lg::ArgInfo>>(m_graph, "ArgInfo")
      .def_readonly("name", &lg::ArgInfo::name)
      .def_readonly("type", &lg::ArgInfo::type);

  py::class_<lg::ConstantInfo, std::shared_ptr<lg::ConstantInfo>>(
      m_graph, "ConstantInfo")
      .def_readonly("type", &lg::ConstantInfo::type);

  // Sequence extractor
  py::module m_seq = m.def_submodule("seq");

  py::class_<ls::ExtractionInfo, std::shared_ptr<ls::ExtractionInfo>>(
      m_seq, "ExtractionInfo")
      .def("accept", &ls::ExtractionInfo::accept)
      .def_readonly("functionInfos", &ls::ExtractionInfo::functionInfos);

  py::class_<ls::FunctionInfo, std::shared_ptr<ls::FunctionInfo>>(
      m_seq, "FunctionInfo")
      .def("accept", &ls::FunctionInfo::accept)
      .def_readonly("name", &ls::FunctionInfo::name)
      .def_readonly("signature", &ls::FunctionInfo::signature)
      .def_readonly("basicBlocks", &ls::FunctionInfo::basicBlocks)
      .def_readonly("str", &ls::FunctionInfo::str);

  py::class_<ls::BasicBlockInfo, std::shared_ptr<ls::BasicBlockInfo>>(
      m_seq, "BasicBlockInfo")
      .def_readonly("name", &ls::BasicBlockInfo::name)
      .def_readonly("instructions", &ls::BasicBlockInfo::instructions);

  py::class_<ls::InstructionInfo, std::shared_ptr<ls::InstructionInfo>>(
      m_seq, "InstructionInfo")
      .def_readonly("tokens", &ls::InstructionInfo::tokens);
}

PYBIND11_MODULE(extractors, m) {
  m.attr("LLVM_VERSION") = LLVM_VERSION_STRING;

  py::class_<IVisitor, PyVisitor>(m, "Visitor").def(py::init<>());

  registerClangDriver(m);

  registerClangExtractor(m);
  registerLLVMExtractor(m);
}
