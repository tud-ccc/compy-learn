import os

import urllib.request
import zipfile

from appdirs import user_data_dir
from git import Repo


class Dataset(object):
    def __init__(self):
        self.name = self.__class__.__name__

        app_dir = user_data_dir(appname="compy-Learn", version="1.0")
        self.dataset_dir = os.path.join(app_dir, self.name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.content_dir = os.path.join(self.dataset_dir, "content")

    def preprocess(self, builder, visitor):
        raise NotImplementedError

    def download_http_and_extract(self, url):
        archive_file = os.path.join(self.dataset_dir, "content.zip")

        if not (os.path.isfile(archive_file) or os.path.isdir(self.content_dir)):
            urllib.request.urlretrieve(url, archive_file)

            os.makedirs(self.content_dir, exist_ok=True)
            with zipfile.ZipFile(archive_file, "r") as f:
                f.extractall(self.content_dir)

        return archive_file, self.content_dir

    def clone_git(self, uri):
        if not os.path.isdir(self.content_dir):
            Repo.clone_from(uri, self.content_dir)

        return self.content_dir
