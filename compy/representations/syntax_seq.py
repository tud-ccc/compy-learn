from compy.representations.extractors import clang_driver_scoped_options
from compy.representations.extractors.extractors import Visitor
from compy.representations.extractors.extractors import ClangDriver
from compy.representations.extractors.extractors import ClangExtractor
from compy.representations.extractors.extractors import clang
from compy.representations import common


class SyntaxSeqVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        if isinstance(v, clang.seq.TokenInfo):
            self.S.append(v.name)


class SyntaxTokenkindVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        if isinstance(v, clang.seq.TokenInfo):
            self.S.append(v.kind)


class SyntaxTokenkindVariableVisitor(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.S = []

    def visit(self, v):
        if isinstance(v, clang.seq.TokenInfo):
            if v.kind == "raw_identifier" and "var" in v.name:
                self.S.append(v.name)
            elif (
                v.name in ["for", "while", "do", "if", "else", "return"]
                or v.name in ["fn_0"]
                or v.name.startswith("int")
                or v.name.startswith("float")
            ):
                self.S.append(v.name)
            else:
                self.S.append(v.kind)


class SyntaxSeqBuilder(common.RepresentationBuilder):
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

    def string_to_info(self, src, additional_include_dir=None, filename=None):
        with clang_driver_scoped_options(self.__clang_driver, additional_include_dir=additional_include_dir, filename=filename):
            return self.__extractor.SeqFromString(src)

    def info_to_representation(self, info, visitor=SyntaxTokenkindVariableVisitor):
        vis = visitor()
        info.accept(vis)

        for token in vis.S:
            if token not in self._tokens:
                self._tokens[token] = 1
            self._tokens[token] += 1

        return common.Sequence(vis.S, self.get_tokens())
