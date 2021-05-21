import pytest

from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import ClangExtractor
from compy.representations.extractors.extractors import LLVMIRExtractor
from compy.representations.extractors.extractors import clang
from compy.representations.extractors.extractors import llvm


program_1fn_1 = """
int foo() {
  return 1;
}
"""

program_1fn_2 = """
int bar(int a) {
  if (a > 10)
    return a;
  return -1;
}
"""

program_1fn_3 = """
int bar(int a) {
  while (a > 10)
    a++;
  return a;
}
"""

program_1fn_4 = """
int bar(int a) {
  int i = 0;
  return a + i;
}
"""

program_2fn = """
int max(int a, int b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}
int foo(int x) {
  return max(1, x);
}
"""


@pytest.fixture
def llvm_extractor_fixture():
    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O0,
        [],
        ["-Wall"],
    )
    llvm_extractor = LLVMIRExtractor(clang_driver)
    return llvm_extractor


@pytest.fixture
def clang_extractor_fixture():
    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.C,
        ClangDriver.OptimizationLevel.O0,
        [],
        ["-Wall"],
    )
    clang_extractor = ClangExtractor(clang_driver)
    return clang_extractor


def print_seq_info(info):
    for functionInfo in info.functionInfos:
        print(functionInfo.str)

        print("Function", functionInfo.name)
        print(" ", functionInfo.signature)
        for bb in functionInfo.basicBlocks:
            print(bb.name)
            for instr in bb.instructions:
                print(" ", instr.tokens)


# LLVM
# ############################

# Graph tests: General
def test_llvm_graph_from_several_functions(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_1)
    assert len(info.functionInfos) == 1

    info2 = llvm_extractor_fixture.GraphFromString(program_2fn)
    assert len(info2.functionInfos) == 2


# Graph tests: Visitors
def test_llvm_graph_visitor(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)

    class TestVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.instructions = []

        def visit(self, v):
            if isinstance(v, llvm.graph.InstructionInfo):
                self.instructions.append(v.opcode)

    visitor = TestVisitor()
    info.accept(visitor)

    assert visitor.instructions[0:5] == ["alloca", "alloca", "store", "load", "icmp"]


# Graph tests: Functions
def test_llvm_graph_functions_have_types_and_names(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_2fn)

    assert [x.name for x in info.functionInfos] == ["max", "foo"]
    assert [x.type for x in info.functionInfos] == ["i32", "i32"]


def test_llvm_graph_functions_have_args_and_basicblocks(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_2fn)
    fn = info.functionInfos[0]

    assert len(fn.args) == 2
    assert len(fn.basicBlocks) > 0


def test_llvm_graph_functions_have_calls(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_2fn)
    fn_0 = info.functionInfos[0]
    fn_1 = info.functionInfos[1]

    bb_entry = fn_1.basicBlocks[0]
    instr_call = bb_entry.instructions[3]

    assert instr_call.callTarget == fn_0.name


# Graph tests: Args
def test_llvm_graph_args_have_types_and_names(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]

    assert fn.args[0].type == "i32"
    assert fn.args[0].name == "val4"


def test_llvm_graph_args_are_referenced_by_operands(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]
    bb_entry = fn.basicBlocks[0]

    arg_0 = fn.args[0]
    instr_alloca = bb_entry.instructions[1]
    instr_load = bb_entry.instructions[2]

    assert instr_load.operands == [arg_0, instr_alloca]


# Graph tests: Basicblocks
def test_llvm_graph_basicblocks_have_names(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_2fn)
    fn = info.functionInfos[0]

    assert [bb.name for bb in fn.basicBlocks] == ["val0", "val1", "val3", "val2"]


def test_llvm_graph_basicblocks_have_successors(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]

    bb_entry = fn.basicBlocks[0]
    bb_if_then = fn.basicBlocks[1]
    bb_if_else = fn.basicBlocks[2]
    bb_return = fn.basicBlocks[3]

    assert bb_entry.successors == [bb_if_then, bb_if_else]
    assert bb_if_then.successors == [bb_return]
    assert bb_if_else.successors == [bb_return]


def test_llvm_graph_basicblocks_successors_support_loops(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_3)
    fn = info.functionInfos[0]

    bb_entry = fn.basicBlocks[0]
    bb_while_cond = fn.basicBlocks[1]
    bb_while_body = fn.basicBlocks[2]
    bb_while_end = fn.basicBlocks[3]

    assert bb_entry.successors == [bb_while_cond]
    assert bb_while_cond.successors == [bb_while_body, bb_while_end]
    assert bb_while_body.successors == [bb_while_cond]


def test_llvm_graph_basicblocks_have_instructions(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]
    bb_entry = fn.basicBlocks[0]

    assert len(bb_entry.instructions) > 0


# Graph tests: Instructions
def test_llvm_graph_instructions_have_type_and_opcode(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]
    bb_entry = fn.basicBlocks[0]

    assert [instr.type for instr in bb_entry.instructions] == [
        "i32*",
        "i32*",
        "void",
        "i32",
        "i1",
        "void",
    ]
    assert [instr.opcode for instr in bb_entry.instructions] == [
        "alloca",
        "alloca",
        "store",
        "load",
        "icmp",
        "br",
    ]


def test_llvm_graph_instructions_have_operands(llvm_extractor_fixture):
    info = llvm_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]
    bb_entry = fn.basicBlocks[0]

    instr_alloca = bb_entry.instructions[1]
    instr_load = bb_entry.instructions[3]

    assert instr_load.operands == [instr_alloca]


# Seq tests: General
def test_llvm_seq_from_several_functions(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_1)
    assert len(info.functionInfos) == 1

    info2 = llvm_extractor_fixture.SeqFromString(program_2fn)
    assert len(info2.functionInfos) == 2


# Graph tests: Visitors
def test_llvm_seq_visitor(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_2)

    class TestVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.instructions = []

        def visit(self, v):
            if isinstance(v, llvm.seq.InstructionInfo):
                self.instructions.append(v.tokens)

    visitor = TestVisitor()
    info.accept(visitor)

    assert visitor.instructions[0][0:5] == ["%", "2", " = ", "alloca", " "]


# Seq tests: Functions
def test_llvm_seq_functions_have_signature_and_names(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]

    assert fn.name == "foo"
    assert fn.signature == ["define ", "dso_local ", "i", "32", " ", "@", "foo", "(", ")", " #", "0"]


# Seq tests: Basicblocks
def test_llvm_seq_functions_have_basicblocks(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]

    assert len(fn.basicBlocks) > 0


def test_llvm_seq_basicblocks_have_names(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_2)
    fn = info.functionInfos[0]

    assert [bb.name for bb in fn.basicBlocks] == ["1", "6", "8", "9"]


def test_llvm_seq_basicblocks_have_tokens(llvm_extractor_fixture):
    info = llvm_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]
    bb = fn.basicBlocks[0]

    assert bb.instructions[0].tokens == ["ret", " ", "i", "32", " ", "1"]


# Clang
# ############################
def get_clang_ast_hierarchy(stmt):
    ret = {}
    for child in stmt.ast_relations:
        ret[child.name] = get_clang_ast_hierarchy(child)
    return ret


def get_statements_with_name(stmt, name):
    ret = []
    for child in stmt.ast_relations:
        if child.name == name:
            ret += [child]
        if hasattr(child, 'ast_relations'):
            ret += get_statements_with_name(child, name)

    return ret


# Graph tests: General
def test_clang_graph_from_several_functions(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_1fn_1)
    assert len(info.functionInfos) == 1

    info2 = clang_extractor_fixture.GraphFromString(program_2fn)
    assert len(info2.functionInfos) == 2


# Graph tests: Visitors
def test_clang_graph_visitor(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_1fn_1)

    class TestVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.stmts = []

        def visit(self, v):
            if isinstance(v, clang.graph.StmtInfo):
                self.stmts.append(v.name)

    visitor = TestVisitor()
    info.accept(visitor)

    assert visitor.stmts == ["CompoundStmt", "ReturnStmt", "IntegerLiteral"]


# Graph tests: Functions
def test_clang_graph_functions_have_types_and_names(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_2fn)

    assert [x.name for x in info.functionInfos] == ["max", "foo"]
    assert [x.type for x in info.functionInfos] == ["int (int, int)", "int (int)"]


# Graph tests: Statements
def test_clang_graph_statements_have_ast_relations(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]

    ast_hierarchy = {
        "IfStmt": {
            "BinaryOperator": {
                "ImplicitCastExpr": {"DeclRefExpr": {}},
                "IntegerLiteral": {},
            },
            "ReturnStmt": {"ImplicitCastExpr": {"DeclRefExpr": {}}},
        },
        "ReturnStmt": {"UnaryOperator": {"IntegerLiteral": {}}},
    }
    assert get_clang_ast_hierarchy(fn.entryStmt) == ast_hierarchy


# Graph tests: Arguments
def test_clang_graph_declarations_args_have_name_and_type(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_2fn)
    fn = info.functionInfos[0]
    decl = fn.args[0]

    assert decl.name == "a"
    assert decl.type == "int"


# Graph tests: References
def test_clang_graph_declarations_args_are_referenced(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_1fn_2)
    fn = info.functionInfos[0]
    declRefExprs = get_statements_with_name(fn.entryStmt, "DeclRefExpr")

    for declRefExpr in declRefExprs:
        for ref_relation in declRefExpr.ref_relations:
            assert type(ref_relation) == clang.graph.DeclInfo


def test_clang_graph_declarations_inner_are_referenced(clang_extractor_fixture):
    info = clang_extractor_fixture.GraphFromString(program_1fn_4)
    fn = info.functionInfos[0]
    declRefExprs = get_statements_with_name(fn.entryStmt, "DeclRefExpr")

    for declRefExpr in declRefExprs:
        for ref_relation in declRefExpr.ref_relations:
            assert type(ref_relation) == clang.graph.DeclInfo


# Seq tests: General
def test_clang_seq_from_several_functions(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_1)
    assert len(info.functionInfos) == 1

    info2 = clang_extractor_fixture.SeqFromString(program_2fn)
    assert len(info2.functionInfos) == 2


# Seq tests: Functions
def test_clang_seq_functions_have_names(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]

    assert fn.name == "foo"


def test_clang_seq_functions_have_token_infos(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]

    assert len(fn.tokenInfos) > 0


# Seq tests: Normalization
def test_clang_seq_function_names_are_normalized(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_1)
    fn = info.functionInfos[0]

    assert [tokenInfo.name for tokenInfo in fn.tokenInfos].count("fn_0") == 1


def test_clang_seq_function_names_are_normalized_differently(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_2fn)
    fn_0 = info.functionInfos[0]
    fn_1 = info.functionInfos[1]

    assert [tokenInfo.name for tokenInfo in fn_0.tokenInfos].count("fn_0") == 1
    assert [tokenInfo.name for tokenInfo in fn_1.tokenInfos].count("fn_1") == 1


def test_clang_seq_function_names_are_normalized_and_reference(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_2fn)
    fn_0 = info.functionInfos[0]
    fn_1 = info.functionInfos[1]

    assert [tokenInfo.name for tokenInfo in fn_0.tokenInfos].count("fn_0") == 1
    assert [tokenInfo.name for tokenInfo in fn_1.tokenInfos].count("fn_1") == 1

    assert [tokenInfo.name for tokenInfo in fn_1.tokenInfos].index("fn_1") < [
        tokenInfo.name for tokenInfo in fn_1.tokenInfos
    ].index("fn_0")


def test_clang_seq_variable_names_are_normalized(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_2)
    fn = info.functionInfos[0]

    assert [tokenInfo.name for tokenInfo in fn.tokenInfos].count("a") == 0
    assert [tokenInfo.name for tokenInfo in fn.tokenInfos].count("var_0") == 3


# Seq tests: Visitors
def test_clang_graph_visitor(clang_extractor_fixture):
    info = clang_extractor_fixture.SeqFromString(program_1fn_1)

    class TestVisitor(Visitor):
        def __init__(self):
            Visitor.__init__(self)
            self.tokens = []

        def visit(self, v):
            if isinstance(v, clang.seq.TokenInfo):
                self.tokens.append(v.name)

    visitor = TestVisitor()
    info.accept(visitor)

    assert visitor.tokens == ["int", "fn_0", "(", ")", "{", "return", "1", ";", "}", ""]
