import numpy as np

from sklearn.model_selection import StratifiedKFold

from compy import datasets as D
from compy import models as M
from compy import representations as R
from compy.representations.extractors import ClangDriver


# Load dataset
dataset = D.OpenCLDevmapDataset()

# Explore combinations
combinations = [
    # CGO 20: AST+DF, CDFG
    (R.ASTGraphBuilder, R.ASTDataVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGVisitor, M.GnnPytorchGeomModel),
    # Arxiv 20: ProGraML
    (R.LLVMGraphBuilder, R.LLVMProGraMLVisitor, M.GnnPytorchGeomModel),
    # PACT 17: DeepTune
    (R.SyntaxSeqBuilder, R.SyntaxTokenkindVariableVisitor, M.RnnTfModel),
    # Extra
    (R.ASTGraphBuilder, R.ASTDataCFGVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGCallVisitor, M.GnnPytorchGeomModel),
    (R.LLVMGraphBuilder, R.LLVMCDFGPlusVisitor, M.GnnPytorchGeomModel),
]

for builder, visitor, model in combinations:
    print("Processing %s-%s-%s" % (builder.__name__, visitor.__name__, model.__name__))

    # Build representation
    clang_driver = ClangDriver(
        ClangDriver.ProgrammingLanguage.OpenCL,
        ClangDriver.OptimizationLevel.O3,
        [(x, ClangDriver.IncludeDirType.User) for x in dataset.additional_include_dirs],
        ["-xcl", "-target", "x86_64-pc-linux-gnu"],
    )
    data = dataset.preprocess(builder(clang_driver), visitor)

    # Train and test
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=204)
    split = kf.split(data["samples"], [sample["info"][5] for sample in data["samples"]])
    for train_idx, test_idx in split:
        model = model(num_types=data["num_types"])
        train_summary = model.train(
            list(np.array(data["samples"])[train_idx]),
            list(np.array(data["samples"])[test_idx]),
        )
        print(train_summary)

        break
