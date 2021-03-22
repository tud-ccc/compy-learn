import networkx as nx

from compy.representations.extractors import clang_driver_scoped_options
from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import ClangExtractor
from compy.representations.extractors.extractors import clang
from compy.representations import common


def filter_type(type):
    if "[" in type or "]" in type:
        return "arrayType"
    elif "(" in type or ")" in type:
        return "fnType"
    elif "int" in type:
        return "intType"
    elif "float" in type:
        return "floatType"
    else:
        return "type"


class ASTVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, attr="function")
            for arg in v.args:
                self.G.add_node(arg, attr=("argument", filter_type(arg.type)))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")


class ASTDataVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast", "data"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, attr="function")
            for arg in v.args:
                self.G.add_node(arg, attr=("argument", filter_type(arg.type)))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")
            for ref_rel in v.ref_relations:
                self.G.add_node(ref_rel, attr=(filter_type(ref_rel.type)))
                self.G.add_edge(v, ref_rel, attr="data")


class ASTDataCFGVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast", "cfg", "in", "data"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        if isinstance(v, clang.graph.FunctionInfo):
            self.G.add_node(v, attr="function")
            for arg in v.args:
                self.G.add_node(arg, attr=("argument", filter_type(arg.type)))
                self.G.add_edge(v, arg, attr="ast")

            self.G.add_node(v.entryStmt, attr=(v.entryStmt.name))
            self.G.add_edge(v, v.entryStmt, attr="ast")

            for cfg_b in v.cfgBlocks:
                self.G.add_node(cfg_b, attr="cfg")
                for succ in cfg_b.successors:
                    self.G.add_edge(cfg_b, succ, attr="cfg")
                    self.G.add_node(succ, attr="cfg")
                for stmt in cfg_b.statements:
                    self.G.add_edge(stmt, cfg_b, attr="in")
                    self.G.add_node(stmt, attr=(stmt.name))

        if isinstance(v, clang.graph.StmtInfo):
            for ast_rel in v.ast_relations:
                self.G.add_node(ast_rel, attr=(ast_rel.name))
                self.G.add_edge(v, ast_rel, attr="ast")
            for ref_rel in v.ref_relations:
                self.G.add_node(ref_rel, attr=(filter_type(ref_rel.type)))
                self.G.add_edge(v, ref_rel, attr="data")


class ASTGraphBuilder(common.RepresentationBuilder):
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
        self.__extractor = ClangExtractor(self.__clang_driver)

        self.__graphs = []

    def string_to_info(self, src, additional_include_dir=None, filename=None):
        with clang_driver_scoped_options(self.__clang_driver, additional_include_dir=additional_include_dir, filename=filename):
            return self.__extractor.GraphFromString(src)

    def info_to_representation(self, info, visitor=ASTDataVisitor):
        vis = visitor()
        info.accept(vis)

        for (n, data) in vis.G.nodes(data=True):
            attr = data["attr"]
            if attr not in self._tokens:
                self._tokens[attr] = 1
            self._tokens[attr] += 1

        return common.Graph(vis.G, self.get_tokens(), vis.edge_types)
