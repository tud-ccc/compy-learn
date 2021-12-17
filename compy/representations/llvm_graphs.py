import networkx as nx

from compy.representations.extractors import clang_driver_scoped_options
from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import LLVMIRExtractor
from compy.representations.extractors.extractors import llvm
from compy.representations import common

def add_missing_call_edges(visitor):
    """Add missing call edges.

    #include <stdio.h>

    int main() {
        int x = Fib(10);
        printf("Result: %d\n", x);
    }

    int Fib(int x) {...}
    """
    for instruction, call in visitor.calls.items():
        called_function = (
            visitor.functions[call]
            if call in visitor.functions
            else None
        )
        if called_function:
            visitor.G.add_edge(instruction, called_function.entryInstruction, attr="call")
            for exit in called_function.exitInstructions:
                visitor.G.add_edge(exit, instruction, attr="call")

def has_edge(G, edge1, edge2, attr):
    """Verify if a edge exists."""
    try:
        edges = G.edges(edge1, data=True)
        for e1, e2, att in edges:
            if e2 == edge2 and att['attr'] == attr:
                return True
        return False
    except Exception:
        return False


class LLVMCFGVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))


class LLVMCFGCompactVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

class LLVMCFGCallVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction, attr="call")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))

            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v, called_function.entryInstruction, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget


class LLVMCFGCallCompactVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v.basicBlock, called_function.entryInstruction.basicBlock, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock, v.basicBlock, attr="call")
                else:
                    self.calls[v] = v.callTarget


class LLVMCFGCallCompactSingleVisitor(Visitor):
    """Do not duplicate edges."""
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(self.G, v.basicBlock, called_function.entryInstruction.basicBlock, "call"):
                        self.G.add_edge(v.basicBlock, called_function.entryInstruction.basicBlock, attr="call")
                    for exit in called_function.exitInstructions:
                        if not has_edge(self.G, exit.basicBlock, v.basicBlock, "call"):
                            self.G.add_edge(exit.basicBlock, v.basicBlock, attr="call")
                else:
                    self.calls[v] = v.callTarget


class LLVMCDFGVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")

class LLVMCDFGCompactVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst.basicBlock, memacc.inst.basicBlock, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_edge(operand.basicBlock, v.basicBlock, attr="data")


class LLVMCDFGCompactSingleVisitor(Visitor):
    """Do not duplicate edges."""
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            if not has_edge(self.G, dep.inst.basicBlock, memacc.inst.basicBlock, "mem"):
                                self.G.add_edge(dep.inst.basicBlock, memacc.inst.basicBlock, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    if not has_edge(self.G, operand, v.basicBlock, "data"):
                        self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    if not has_edge(self.G, operand.basicBlock, v.basicBlock, "data"):
                        self.G.add_edge(operand.basicBlock, v.basicBlock, attr="data")


class LLVMCDFGCallVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))

            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v, called_function.entryInstruction, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGCallCompactVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst.basicBlock, memacc.inst.basicBlock, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v.basicBlock, called_function.entryInstruction.basicBlock, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit.basicBlock, v.basicBlock, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                    self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_edge(operand.basicBlock, v.basicBlock, attr="data")


class LLVMCDFGCallCompactSingleVisitor(Visitor):
    """Do not duplicate edges."""
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction.basicBlock, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

            # Memory accesses edges.
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            if not has_edge(self.G, dep.inst.basicBlock, memacc.inst.basicBlock, "mem"):
                                self.G.add_edge(dep.inst.basicBlock, memacc.inst.basicBlock, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG nodes: Inner-BB.
            attr = '_'.join([insn.opcode for insn in v.instructions])
            self.G.add_node(v, attr=attr)

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v, succ.instructions[0].basicBlock, attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v.basicBlock, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    if not has_edge(self.G, v.basicBlock, called_function.entryInstruction.basicBlock, "call"):
                        self.G.add_edge(v.basicBlock, called_function.entryInstruction.basicBlock, attr="call")
                    for exit in called_function.exitInstructions:
                        if not has_edge(self.G, exit.basicBlock, v.basicBlock, "call"):
                            self.G.add_edge(exit.basicBlock, v.basicBlock, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo):
                     if not has_edge(self.G, operand, v.basicBlock, "data"):
                         self.G.add_edge(operand, v.basicBlock, attr="data")
                if isinstance(operand, llvm.graph.InstructionInfo):
                    if not has_edge(self.G, operand.basicBlock, v.basicBlock, "data"):
                        self.G.add_edge(operand.basicBlock, v.basicBlock, attr="data")


class LLVMCDFGPlusVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call", "bb"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function root node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction, attr="cfg")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))
                self.G.add_edge(v, arg, attr="data")

            # Memory accesses
            for memacc in v.memoryAccesses:
                if memacc.inst:
                    for dep in memacc.dependencies:
                        if dep.inst:
                            self.G.add_edge(dep.inst, memacc.inst, attr="mem")

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # BB nodes
            self.G.add_node(v, attr="bb")
            for instr in v.instructions:
                self.G.add_edge(instr, v, attr="bb")
            for succ in v.successors:
                self.G.add_edge(v, succ, attr="bb")

            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))

            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v, v.function, attr="call")

            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v, called_function.entryInstruction, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMProGraMLVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}
        self.calls = {}

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
            self.functions[v.name] = v

            # Function node.
            self.G.add_node(v, attr="function")
            self.G.add_edge(v, v.entryInstruction, attr="call")

            # Function arg nodes.
            for arg in v.args:
                self.G.add_node(arg, attr=(arg.type))

        if isinstance(v, llvm.graph.BasicBlockInfo):
            # CFG edges: Inner-BB.
            instr_prev = v.instructions[0]
            for instr in v.instructions[1:]:
                self.G.add_edge(instr_prev, instr, attr="cfg")
                instr_prev = instr

            # CFG edges: Inter-BB
            for succ in v.successors:
                self.G.add_edge(v.instructions[-1], succ.instructions[0], attr="cfg")

        if isinstance(v, llvm.graph.InstructionInfo):
            # Instruction nodes.
            self.G.add_node(v, attr=(v.opcode))

            # Call edges.
            if v.opcode == "ret":
                self.G.add_edge(v, v.function, attr="call")
            if v.opcode == "call":
                called_function = (
                    self.functions[v.callTarget]
                    if v.callTarget in self.functions
                    else None
                )
                if called_function:
                    self.G.add_edge(v, called_function.entryInstruction, attr="call")
                    for exit in called_function.exitInstructions:
                        self.G.add_edge(exit, v, attr="call")
                else:
                    self.calls[v] = v.callTarget

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.ConstantInfo
                ):
                    self.G.add_node(operand, attr=(operand.type))
                    self.G.add_edge(operand, v, attr="data")
                elif isinstance(operand, llvm.graph.InstructionInfo):
                    self.G.add_node((v, operand), attr=(operand.type))
                    self.G.add_edge(operand, (v, operand), attr="data")
                    self.G.add_edge((v, operand), v, attr="data")


class LLVMGraphBuilder(common.RepresentationBuilder):
    def __init__(self, clang_driver=None):
        common.RepresentationBuilder.__init__(self)

        if clang_driver:
            self.__clang_driver = clang_driver
        else:
            self.__clang_driver = ClangDriver(
                ClangDriver.ProgrammingLanguage.C,
                ClangDriver.OptimizationLevel.O3,
                [],
                ["-Wall"],
            )
        self.__extractor = LLVMIRExtractor(self.__clang_driver)

    def string_to_info(self, src, additional_include_dir=None, filename=None):
        with clang_driver_scoped_options(self.__clang_driver, additional_include_dir=additional_include_dir, filename=filename):
            return self.__extractor.GraphFromString(src)

    def info_to_representation(self, info, visitor=LLVMCDFGVisitor):
        vis = visitor()
        info.accept(vis)

        if 'calls' in vis.__dict__:
            add_missing_call_edges(vis)

        for (n, data) in vis.G.nodes(data=True):
            attr = data["attr"]
            if attr not in self._tokens:
                self._tokens[attr] = 1
            self._tokens[attr] += 1

        return common.Graph(vis.G, self.get_tokens(), vis.edge_types)
