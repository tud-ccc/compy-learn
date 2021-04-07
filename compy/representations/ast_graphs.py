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


def add_ast_edges(g: nx.MultiDiGraph, node):
    """Add edges with attr `ast` that represent the AST parent-child relationship"""

    if isinstance(node, clang.graph.FunctionInfo):
        g.add_node(node, attr="function")
        for arg in node.args:
            g.add_node(arg, attr=("argument", filter_type(arg.type)))
            g.add_edge(node, arg, attr="ast")

        g.add_node(node.entryStmt, attr=(node.entryStmt.name))
        g.add_edge(node, node.entryStmt, attr="ast")

    if isinstance(node, clang.graph.StmtInfo):
        for ast_rel in node.ast_relations:
            g.add_node(ast_rel, attr=(ast_rel.name))
            g.add_edge(node, ast_rel, attr="ast")


def add_ref_edges(g: nx.MultiDiGraph, node):
    """Add edges with attr `data` for data references of the given node"""

    if isinstance(node, clang.graph.StmtInfo):
        for ref_rel in node.ref_relations:
            g.add_node(ref_rel, attr=(filter_type(ref_rel.type)))
            g.add_edge(node, ref_rel, attr="data")


def add_cfg_edges(g: nx.MultiDiGraph, node):
    """Add edges with attr `cfg` or `in` for control flow for the given node"""

    if isinstance(node, clang.graph.FunctionInfo):
        for cfg_b in node.cfgBlocks:
            g.add_node(cfg_b, attr="cfg")
            for succ in cfg_b.successors:
                g.add_edge(cfg_b, succ, attr="cfg")
                g.add_node(succ, attr="cfg")
            for stmt in cfg_b.statements:
                g.add_edge(stmt, cfg_b, attr="in")
                g.add_node(stmt, attr=(stmt.name))


def add_token_ast_edges(g: nx.MultiDiGraph, node):
    """Add edges with attr `token` connecting tokens to the closest AST node covering them"""
    if hasattr(node, 'tokens'):
        for token in node.tokens:
            g.add_node(token, attr=token.name, seq_order=token.index)
            g.add_edge(token, node, attr="token")


class ASTVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        add_ast_edges(self.G, v)


class ASTDataVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast", "data"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        add_ast_edges(self.G, v)
        add_ref_edges(self.G, v)


class ASTDataCFGVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast", "cfg", "in", "data"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        add_ast_edges(self.G, v)
        add_ref_edges(self.G, v)
        add_cfg_edges(self.G, v)


class ASTDataCFGTokenVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.edge_types = ["ast", "cfg", "in", "data", "token"]
        self.G = nx.MultiDiGraph()

    def visit(self, v):
        add_ast_edges(self.G, v)
        add_ref_edges(self.G, v)
        add_cfg_edges(self.G, v)
        add_token_ast_edges(self.G, v)


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
