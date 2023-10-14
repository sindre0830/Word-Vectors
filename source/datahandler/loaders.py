import gensim.downloader
import itertools
import nltk
import nltk.corpus
import tqdm
import numpy as np
import collections


class Corpus():
    def __init__(self) -> None:
        self.sentences = None
        self.words = None

    def build(self, pipeline) -> None:
        for step in tqdm.tqdm(pipeline, desc="Building corpus"):
            step()

    def download(self) -> None:
        self.sentences = gensim.downloader.load("text8")

    def flatten(self) -> None:
        self.words = list(itertools.chain.from_iterable(self.sentences))

    def filter_stop_words(self) -> None:
        nltk.download("stopwords", quiet=True)
        stop_words = nltk.corpus.stopwords.words("english")
        self.corpus = [[word for word in sentence if word not in stop_words] for sentence in self.sentences]


class Vocabulary():
    def __init__(self, add_padding: bool, add_unknown: bool) -> None:
        self.token_to_index = {}
        self.index_to_token = {}
        self.token_freq = {}
        self.total_words = 0

        self.padding_token = "<PAD>"
        self.unknown_token = "<UNK>"
        self.padding_index = None
        self.unknown_index = None

        if add_padding:
            self.padding_index = self.add_token(self.padding_token)
        if add_unknown:
            self.unknown_index = self.add_token(self.unknown_token)
    
    def build(self, words: list[str], size: int):
        word_freqs = collections.Counter(words)
        common_words = word_freqs.most_common(n=size)
        for word, freq in tqdm.tqdm(common_words, desc="Building vocabulary"):
            self.add_token(word, freq)

    def add_token(self, token: str, freq=1) -> int:
        if token not in self.token_to_index:
            idx = len(self.token_to_index)
            self.token_to_index[token] = idx
            self.index_to_token[idx] = token
            self.token_freq[token] = freq
        else:
            self.token_freq[token] += freq
        self.total_words += freq
        return self.get_index(token)

    def get_index(self, token: str, default: int = None):
        if default is None:
            default = self.unknown_index
        return self.token_to_index.get(token, default)

    def get_token(self, index: int, default: str = None) -> str:
        if default is None:
            default = self.unknown_token
        return self.index_to_token.get(index)

    def get_frequency(self, token: str, default=0) -> int:
        return self.token_freq.get(token, default)

    def subsample_probability(self, token: str, threshold=1e-5):
        """Compute the probability of keeping the given token."""
        freq_ratio = self.get_frequency(token) / self.total_words
        return 1 - np.sqrt(threshold / freq_ratio)

    def __len__(self):
        return len(self.token_to_index)
