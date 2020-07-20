import os

from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import llvm
from compy.representations.llvm_seq import LLVMSeqBuilder
from compy.representations.llvm_seq import LLVMSeqVisitor


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
    builder = LLVMSeqBuilder()
    info = builder.string_to_info(program_1fn_2)
    seq = builder.info_to_representation(info)


# General tests: Plot
def test_plot(tmpdir):
    builder = LLVMSeqBuilder()
    info = builder.string_to_info(program_1fn_2)
    seq = builder.info_to_representation(info)

    outfile = os.path.join(tmpdir, "syntax_seq.png")
    seq.draw(path=outfile, width=8)

    assert os.path.isfile(outfile)

    # os.system('xdg-open ' + str(outfile))
