import os
import pytest

import networkx as nx

from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import llvm
from compy.representations.llvm_graphs import LLVMGraphBuilder
from compy.representations.llvm_graphs import LLVMCDFGVisitor
from compy.representations.llvm_graphs import LLVMCDFGCallVisitor
from compy.representations.llvm_graphs import LLVMCDFGPlusVisitor
from compy.representations.llvm_graphs import LLVMProGraMLVisitor


program_1fn_2 = """
int bar(int a) {
  if (a > 10)
    return a;
  return -1;
}
"""

program_fib = """
int fib(int x) {
    switch(x) {
        case 0:
            return 0;
        case 1:
            return 1;
        default:
            return fib(x-1) + fib(x-2);
    }
}
"""


def get_nodes_with_attr(graph, attr):
    return [x for x, y in graph.G.nodes(data=True) if y["attr"] == attr]


def get_function_nodes_by_name(graph):
    ret = {}
    for function_node in get_nodes_with_attr(graph, "function"):
        ret[function_node.name] = function_node

    return ret


def get_first_instruction(graph):
    for node in graph.G.nodes():
        if isinstance(node, llvm.graph.InstructionInfo):
            return node


def get_all_instructions(graph):
    ret = []
    for node in graph.G.nodes():
        if isinstance(node, llvm.graph.InstructionInfo):
            ret.append(node)
    return ret


def explore_cfg_with_dfs(graph, start_node):
    to_explore = [start_node]
    explored = []
    while len(to_explore):
        node = to_explore.pop(0)
        explored.append(node)

        for u, v, data in graph.G.out_edges(node, data=True):
            if data["attr"] == "cfg" and data not in explored:
                to_explore.append(v)

    return explored


# General tests: Construction
def test_construct_with_custom_visitor():
    class CustomVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.edge_types = []
            self.G = nx.DiGraph()

        def visit(self, v):
            if not isinstance(v, llvm.graph.ExtractionInfo):
                self.G.add_node(v, attr=type(v))

    builder = LLVMGraphBuilder()
    info = builder.string_to_info(program_1fn_2)
    graph = builder.info_to_representation(info, CustomVisitor)

    assert len(graph.G) > 0


# General tests: Attributes
def test_get_node_list():
    builder = LLVMGraphBuilder()
    info = builder.string_to_info(program_1fn_2)
    graph = builder.info_to_representation(info, LLVMCDFGVisitor)
    nodes = graph.get_node_list()

    assert len(nodes) > 0


def test_get_edge_list():
    builder = LLVMGraphBuilder()
    info = builder.string_to_info(program_1fn_2)
    graph = builder.info_to_representation(info, LLVMCDFGVisitor)
    edges = graph.get_edge_list()

    assert len(edges) > 0

    assert type(edges[0][0]) is int
    assert type(edges[0][1]) is int
    assert type(edges[0][2]) is int


# General tests: Plot
def test_plot(tmpdir):
    for visitor in [LLVMCDFGVisitor, LLVMCDFGPlusVisitor, LLVMProGraMLVisitor]:
        builder = LLVMGraphBuilder()
        info = builder.string_to_info(program_fib)
        graph = builder.info_to_representation(info, visitor)

        outfile = os.path.join(tmpdir, str(visitor.__name__) + ".png")
        graph.draw(path=outfile, with_legend=True)

        assert os.path.isfile(outfile)

    # os.system('xdg-open ' + str(tmpdir))


# All visitors
def test_all_visitors():
    for visitor in [
        LLVMCDFGVisitor,
        LLVMCDFGCallVisitor,
        LLVMCDFGPlusVisitor,
        LLVMProGraMLVisitor,
    ]:
        builder = LLVMGraphBuilder()
        info = builder.string_to_info(program_1fn_2)
        ast = builder.info_to_representation(info, visitor)

        assert ast


# CDFG
# ############################
@pytest.fixture
def llvm_cdfg_graph():
    builder = LLVMGraphBuilder()
    info = builder.string_to_info(program_fib)

    return builder.info_to_representation(info, LLVMCDFGVisitor)


# CFG edges
def d_test_cdfg_cfg_edges_reach_all_nodes(llvm_cdfg_graph):
    first_instruction = get_first_instruction(llvm_cdfg_graph)
    explored = explore_cfg_with_dfs(llvm_cdfg_graph, first_instruction)

    assert set(explored) == set(get_all_instructions(llvm_cdfg_graph))


# ProGraML
# ############################
@pytest.fixture
def llvm_programl_graph():
    builder = LLVMGraphBuilder()
    info = builder.string_to_info(program_fib)

    return builder.info_to_representation(info, LLVMProGraMLVisitor)


# General
def test_programl_has_root_node(llvm_programl_graph):
    assert llvm_programl_graph.get_node_str_list().count("function") == 1


# Call edges
def test_programl_call_edges_exist_from_ret_instructions_to_root_node(
    llvm_programl_graph,
):
    for ret_instr in get_nodes_with_attr(llvm_programl_graph, "ret"):
        assert llvm_programl_graph.G.has_edge(ret_instr, ret_instr.function) == True


def test_programl_call_edges_exist_from_call_instructions_to_entry_instructions(
    llvm_programl_graph,
):
    function_nodes_by_name = get_function_nodes_by_name(llvm_programl_graph)

    for call_instr in get_nodes_with_attr(llvm_programl_graph, "call"):
        called_function_node = function_nodes_by_name[call_instr.callTarget]

        assert llvm_programl_graph.G.has_edge(
            call_instr, called_function_node.entryInstruction
        )


def test_programl_call_edges_exist_from_exit_instructions_to_their_callsite_instructions(
    llvm_programl_graph,
):
    function_nodes_by_name = get_function_nodes_by_name(llvm_programl_graph)

    for call_instr in get_nodes_with_attr(llvm_programl_graph, "call"):
        called_function_node = function_nodes_by_name[call_instr.callTarget]

        for exit_instruction in called_function_node.exitInstructions:
            assert llvm_programl_graph.G.has_edge(exit_instruction, call_instr)


# CFG edges
def test_programl_cfg_edges_reach_all_nodes(llvm_programl_graph):
    first_instruction = get_first_instruction(llvm_programl_graph)
    explored = explore_cfg_with_dfs(llvm_programl_graph, first_instruction)

    assert set(explored) == set(get_all_instructions(llvm_programl_graph))
