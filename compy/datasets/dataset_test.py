import os
import pytest
import shutil

from compy.datasets import dataset


@pytest.fixture
def dataset_fixture():
    class TestDataset(dataset.Dataset):
        def __init__(self):
            super().__init__()

    ds = TestDataset()

    yield ds

    shutil.rmtree(ds.dataset_dir)


def test_dataset_has_correct_name(dataset_fixture):
    assert dataset_fixture.name == "TestDataset"


def test_app_dir_is_initialized(dataset_fixture):
    assert type(dataset_fixture.dataset_dir) is str


def test_app_dir_exists(dataset_fixture):
    assert os.path.isdir(dataset_fixture.dataset_dir)


def test_download_http_and_extract(dataset_fixture):
    zip_file, content_dir = dataset_fixture.download_http_and_extract(
        "http://wwwpub.zih.tu-dresden.de/~s9602232/test.zip"
    )

    assert os.path.isfile(zip_file)
    assert os.path.isdir(content_dir)


def test_clone_git(dataset_fixture):
    content_dir = dataset_fixture.clone_git(
        "https://github.com/alexanderb14/build-webrtc-builds.git"
    )

    assert os.path.isdir(content_dir)
