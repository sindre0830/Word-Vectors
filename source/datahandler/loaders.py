from constants import (
    PROJECT_DIRECTORY_PATH
)
from utils import (
    save_numpy,
    load_numpy
)

import os
import gensim.downloader
import itertools
import nltk
import nltk.corpus
import tqdm
import numpy as np
import collections
import torch


class Corpus():
    def __init__(self):
        self.sentences = None
        self.words = None

    def build(self, pipeline) -> None:
        for step in tqdm.tqdm(pipeline, desc="Building corpus"):
            step()

    def download(self) -> None:
        self.sentences = list(gensim.downloader.load("text8"))

    def flatten(self) -> None:
        self.words = list(itertools.chain.from_iterable(self.sentences))

    def filter_stop_words(self) -> None:
        nltk.download("stopwords", quiet=True)
        stop_words = nltk.corpus.stopwords.words("english")
        self.corpus = [[word for word in sentence if word not in stop_words] for sentence in self.sentences]


class Vocabulary():
    def __init__(self, add_padding: bool, add_unknown: bool):
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

    def build(self, words: list[str], size: int) -> None:
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

    def get_index(self, token: str, default: int = None) -> int:
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

    def __contains__(self, token: str):
        return token in self.token_to_index


class DataLoaderCBOW():
    def __init__(self, batch_size: int):
        self.context_words = None
        self.target_words = None

        self._num_samples = 0
        self._batch_size = batch_size

    def build(self, sentences: list[list[str]], vocabulary: Vocabulary, window_size: int, device: str):
        context_words_filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "cbow", "training_data", "context_words.npy")
        target_words_filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "cbow", "training_data", "target_words.npy")

        if os.path.exists(context_words_filepath) and os.path.exists(target_words_filepath):
            progress_bar = tqdm.tqdm(desc="Building training data", total=1)
            self.context_words = torch.tensor(load_numpy(context_words_filepath), dtype=torch.long, device=device)
            self.target_words = torch.tensor(load_numpy(target_words_filepath), dtype=torch.long, device=device)
            self._num_samples = len(self.target_words)
            progress_bar.update(1)
            return

        context_words = []
        target_words = []
        for sentence in tqdm.tqdm(sentences, desc="Building training data"):
            for center_position, center_word in enumerate(sentence):
                if center_word not in vocabulary:
                    continue
                # define the boundaries of the window
                start_position = max(0, center_position - window_size)
                end_position = min(len(sentence), center_position + window_size + 1)
                # extract words around the center word within the window
                context = [
                    vocabulary.get_index(word, vocabulary.padding_index)
                    for pos, word in enumerate(sentence[start_position:end_position])
                    if pos != center_position - start_position
                ]
                # add padding index if context words are missing
                padding_needed = 2 * window_size - len(context)
                context.extend([vocabulary.padding_index] * padding_needed)

                context_words.append(context)
                target_words.append(vocabulary.get_index(center_word))

        context_words = np.array(context_words)
        target_words = np.array(target_words)
        save_numpy(context_words_filepath, context_words)
        save_numpy(target_words_filepath, target_words)

        self.context_words = torch.tensor(context_words, dtype=torch.long, device=device)
        self.target_words = torch.tensor(target_words, dtype=torch.long, device=device)
        self._num_samples = len(self.target_words)

    def __iter__(self):
        for start in range(0, self._num_samples, self._batch_size):
            end = min(start + self._batch_size, self._num_samples)
            yield (self.context_words[start:end], self.target_words[start:end])

    def __len__(self):
        return (self._num_samples + self._batch_size - 1) // self._batch_size
