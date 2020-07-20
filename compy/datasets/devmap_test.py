import pytest

from compy.datasets import OpenCLDevmapDataset
from compy.representations import RepresentationBuilder


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


class TestBuilder(RepresentationBuilder):
    def string_to_info(self, src, additional_include_dir):
        functionInfo = objectview({"name": "xyz"})
        return objectview({"functionInfos": [functionInfo]})

    def info_to_representation(self, info, visitor):
        return "Repr"


@pytest.fixture
def devmap_fixture():
    ds = OpenCLDevmapDataset()
    yield ds


def d_test_preprocess(devmap_fixture):
    builder = TestBuilder()
    devmap_fixture.preprocess(builder, None)
