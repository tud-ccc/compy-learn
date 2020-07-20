import os

from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import clang
from compy.representations.syntax_seq import SyntaxSeqBuilder
from compy.representations.syntax_seq import SyntaxSeqVisitor
from compy.representations.syntax_seq import SyntaxTokenkindVisitor
from compy.representations.syntax_seq import SyntaxTokenkindVariableVisitor


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


# Construction
def test_construct_with_custom_visitor():
    builder = SyntaxSeqBuilder()
    info = builder.string_to_info(program_1fn_2)
    seq = builder.info_to_representation(info, SyntaxTokenkindVariableVisitor)


# General tests: Plot
def test_plot(tmpdir):
    builder = SyntaxSeqBuilder()
    info = builder.string_to_info(program_1fn_2)
    seq = builder.info_to_representation(info)

    outfile = os.path.join(tmpdir, "syntax_seq.png")
    seq.draw(path=outfile, width=8)

    assert os.path.isfile(outfile)

    # os.system('xdg-open ' + str(outfile))


# All visitors
def test_all_visitors():
    for visitor in [
        SyntaxSeqVisitor,
        SyntaxTokenkindVisitor,
        SyntaxTokenkindVariableVisitor,
    ]:
        builder = SyntaxSeqBuilder()
        info = builder.string_to_info(program_1fn_2)
        ast = builder.info_to_representation(info, visitor)

        assert ast
