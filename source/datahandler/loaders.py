import utils

import gensim.downloader
import itertools


class CorpusLoader():
    def __init__(self) -> None:
        self.corpus = None
        self.is_flattened = False

    def download(self) -> None:
        self.corpus = gensim.downloader.load("text8")

    def flatten(self) -> None:
        self.corpus = list(itertools.chain.from_iterable(self.corpus))
        self.is_flat = True
