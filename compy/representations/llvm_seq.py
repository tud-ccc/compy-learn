from compy.representations.extractors import clang_driver_scoped_options
from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import LLVMIRExtractor
from compy.representations.extractors.extractors import llvm
from compy.representations import common


def merge_after_element_on_condition(elements, element_conditions):
    """
    Ex.: If merged on conditions ['a'], ['a', 'b', 'c', 'a', 'e'] becomes ['ab', 'c', 'ae']
    """
    for i in range(len(elements) - 2, -1, -1):
        if elements[i] in element_conditions:
            elements[i] = elements[i] + elements.pop(i + 1)

    return elements


def filer_elements(elements, element_filter):
    """
    Ex.: If filtered on elements [' '], ['a', ' ', 'c'] becomes ['a', 'c']
    """
    return [element for element in elements if element not in element_filter]


def strip_elements(elements, element_filters):
    """
    Ex.: If stripped on elments [' '], ['a', ' b', 'c'] becomes ['a', 'b', 'c']
    """
    ret = []
    for element in elements:
        for element_filter in element_filters:
            element = element.strip(element_filter)
        ret.append(element)

    return ret


def strip_function_name(elements):
    for i in range(len(elements) - 1):
        if elements[i] == "@":
            elements[i + 1] = "fn_0"

    return elements


def transform_elements(elements):
    elements = merge_after_element_on_condition(elements, ["%", "i"])
    elements = strip_elements(elements, ["\n", " "])
    elements = filer_elements(elements, ["", " ", "local_unnamed_addr"])

    return elements


class LLVMSeqVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        if isinstance(v, llvm.seq.FunctionInfo):
            self.S += strip_function_name(transform_elements(v.signature))

        if isinstance(v, llvm.seq.BasicBlockInfo):
            self.S += [v.name + ":"]

        if isinstance(v, llvm.seq.InstructionInfo):
            self.S += transform_elements(v.tokens)


class LLVMSeqBuilder(common.RepresentationBuilder):
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
            return self.__extractor.SeqFromString(src)
        
    def info_to_representation(self, info, visitor=LLVMSeqVisitor):
        vis = visitor()
        info.accept(vis)

        for token in vis.S:
            if token not in self._tokens:
                self._tokens[token] = 1
            self._tokens[token] += 1

        return common.Sequence(vis.S, self.get_tokens())
