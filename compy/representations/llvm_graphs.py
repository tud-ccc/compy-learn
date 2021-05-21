import networkx as nx

from compy.representations.extractors import clang_driver_scoped_options
from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import LLVMIRExtractor
from compy.representations.extractors.extractors import llvm
from compy.representations import common


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


class LLVMCDFGCallVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call"]
        self.G = nx.MultiDiGraph()
        self.functions = {}

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

            # Operands.
            for operand in v.operands:
                if isinstance(operand, llvm.graph.ArgInfo) or isinstance(
                    operand, llvm.graph.InstructionInfo
                ):
                    self.G.add_edge(operand, v, attr="data")


class LLVMCDFGPlusVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["cfg", "data", "mem", "call", "bb"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, llvm.graph.FunctionInfo):
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

        for (n, data) in vis.G.nodes(data=True):
            attr = data["attr"]
            if attr not in self._tokens:
                self._tokens[attr] = 1
            self._tokens[attr] += 1

        return common.Graph(vis.G, self.get_tokens(), vis.edge_types)
