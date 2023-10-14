import utils

import gensim.downloader


class CorpusLoader():
    def __init__(self) -> None:
        self.corpus = None
        self.is_flattened = False

    def download(self) -> None:
        self.corpus = gensim.downloader.load("text8")
