from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils

import os
import gensim.downloader
import itertools
import nltk
import nltk.corpus
import tqdm
import numpy as np
import collections
import torch
import zipfile
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats


class Corpus():
    def __init__(self):
        self.sentences = None
        self.words = None

    def build(self, pipeline, data_directory: str) -> None:
        for step in tqdm.tqdm(pipeline, desc="Building corpus"):
            step()
        if self.words is not None:
            utils.plot_frequency_distribution(self.words, data_directory)

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
        size -= len(self)
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

    def subsample(self, token: str, threshold=1e-5) -> bool:
        prob = (np.sqrt(self.get_frequency(token) / (threshold * self.total_words)) + 1) * (threshold * self.total_words) / self.get_frequency(token)
        return (prob < np.random.rand())

    def __len__(self):
        return len(self.token_to_index)

    def __contains__(self, token: str):
        return token in self.token_to_index


class ValidationLoader():
    def __init__(self, data_directory: str):
        self.data_directory = data_directory

        self.analogy_test: np.ndarray = None
        self.analogy_similarity_rank: np.ndarray = None

        self.word_pair_similarity_test: np.ndarray = None
        self.word_pair_similarity_human_scores: np.ndarray = None
        self.word_pair_similarity_model_scores: np.ndarray = None

    def build(self, vocabulary: Vocabulary):
        progress_bar = tqdm.tqdm(desc="Building validation data", total=2)
        # get analogy test set
        filepath_cache = os.path.join(PROJECT_DIRECTORY_PATH, "data", self.data_directory, "validation_data", "analogy_test.npy")
        if os.path.exists(filepath_cache):
            # load cache
            self.analogy_test = utils.load_numpy(filepath_cache)
        else:
            analogies = []
            # download raw data
            filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "analogy_test.txt")
            utils.download_file("http://download.tensorflow.org/data/questions-words.txt", filepath)
            with open(filepath, "r") as file:
                for line in file:
                    # skip headers
                    if line.startswith(":"):
                        continue
                    # tokenize words
                    words = line.strip().lower().split()
                    # convert to words to their index and check that all the words are in the vocabulary
                    test = [
                        vocabulary.get_index(word)
                        for word in words
                        if word in vocabulary
                    ]
                    if len(test) != 4:
                        continue
                    analogies.append(test)
            # save to cache
            self.analogy_test = np.array(analogies)
            utils.save_numpy(filepath_cache, self.analogy_test)
        progress_bar.update(1)
        # get wordsim353 test set
        filepath_cache = os.path.join(PROJECT_DIRECTORY_PATH, "data", self.data_directory, "validation_data", "wordsim353_test.npy")
        if os.path.exists(filepath_cache):
            # load cache
            self.word_pair_similarity_test = utils.load_numpy(filepath_cache)
        else:
            word_pairs = []
            # download raw data
            filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "wordsim353_test", "combined.csv")
            if not os.path.exists(filepath):
                filepath_zipped = os.path.join(PROJECT_DIRECTORY_PATH, "data", "wordsim353_test.zip")
                utils.download_file("https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip", filepath_zipped)
                with zipfile.ZipFile(filepath_zipped, "r") as file:
                    file.extractall(os.path.dirname(filepath))
            # parse raw data
            with open(filepath, "r") as file:
                reader = csv.reader(file, delimiter="\t")
                # skip header
                next(reader)
                for row in reader:
                    # tokenize elements and skip unfinished rows
                    elements = row[0].lower().split(',')
                    if len(elements) != 3:
                        continue
                    # convert to words to their index and check that all the words are in the vocabulary
                    word1, word2, sim_score = elements
                    if word1 not in vocabulary or word2 not in vocabulary:
                        continue
                    word_pairs.append([float(vocabulary.get_index(word1)), float(vocabulary.get_index(word2)), float(sim_score)])
            # save to cache
            self.word_pair_similarity_test = np.array(word_pairs)
            utils.save_numpy(filepath_cache, self.word_pair_similarity_test)
        progress_bar.update(1)

    def evaluate_analogies(self, embeddings: np.ndarray, quiet=False):
        similarity_rank = []
        for word1_idx, word2_idx, word3_idx, word4_idx in tqdm.tqdm(self.analogy_test, desc="Evaluating Google Analogy test", disable=quiet):
            # get vector representations
            word_vector_1 = embeddings[word1_idx]
            word_vector_2 = embeddings[word2_idx]
            word_vector_3 = embeddings[word3_idx]
            # compute the analogy vector
            analogy_vector = word_vector_1 - word_vector_2
            predicted_vector = utils.normalize(word_vector_3 + analogy_vector)
            # get cosine similarity scores
            cosine_similarities: np.ndarray = utils.cosine_similarity(embeddings, predicted_vector)
            # exclude input words from similarity scores
            cosine_similarities[word1_idx] = -float("inf")
            cosine_similarities[word2_idx] = -float("inf")
            cosine_similarities[word3_idx] = -float("inf")
            # get rank of word4 in similarity scores
            sorted_indicies = cosine_similarities.argsort()[::-1]
            word4_rank = np.where(sorted_indicies == word4_idx)[0][0]
            similarity_rank.append(word4_rank)
        self.analogy_similarity_rank = np.array(similarity_rank)

    def analogies_accuracy(self, k=5):
        # ensure the ranks array is provided
        if self.analogy_similarity_rank is None:
            raise ValueError("You need to run evaluate_analogies first")

        correct_predictions = self.analogy_similarity_rank[self.analogy_similarity_rank < k]
        total_predictions = len(self.analogy_similarity_rank)
        return len(correct_predictions) / total_predictions

    def plot_analogies_rank(self, k=5):
        # ensure the ranks array is provided
        if self.analogy_similarity_rank is None:
            raise ValueError("You need to run evaluate_analogies first")

        rank_counts = [np.sum(self.analogy_similarity_rank == i) for i in range(k)]

        title = f"Rank Distribution of Correct Analogy"

        _, ax = plt.subplots()
        ax.bar(range(1, k + 1), rank_counts)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Number of Occurrences")
        ax.set_title(title)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(range(1, k + 1))
        plt.grid(axis='y')

        utils.save_plot(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "data", self.data_directory, "plots", title + ".png"))
        plt.close()

    def evaluate_word_pair_similarity(self, embeddings: np.ndarray, quiet=False):
        model_scores = []
        human_scores = []
        #print(self.word_pair_similarity_test)
        for word1_idx, word2_idx, human_score in tqdm.tqdm(self.word_pair_similarity_test, desc="Evaluating WordSim353 test", disable=quiet):
            # get vector representations
            word_vector_1 = embeddings[int(word1_idx)]
            word_vector_2 = embeddings[int(word2_idx)]
            # get cosine similarity
            model_score = utils.cosine_similarity(word_vector_1, word_vector_2)
            model_scores.append(model_score)
            human_scores.append(human_score)
        self.word_pair_similarity_model_scores = np.array(model_scores)
        self.word_pair_similarity_human_scores = np.array(human_scores)

    def word_pair_spearman_correlation(self):
        spearman_correlation_coefficient, _ = scipy.stats.spearmanr(self.word_pair_similarity_model_scores, self.word_pair_similarity_human_scores)
        return spearman_correlation_coefficient

    def plot_word_pair_similarity(self):
        title = "WordSim-353 Evaluation"
        plt.scatter(self.word_pair_similarity_human_scores, self.word_pair_similarity_model_scores, alpha=0.6, edgecolors="w", linewidth=0.5)
        plt.title(title)
        plt.xlabel("Human Judgement Scores")
        plt.ylabel("Model Cosine Similarity")
        # line of best fit
        m, b = np.polyfit(self.word_pair_similarity_human_scores, self.word_pair_similarity_model_scores, 1)
        plt.plot(self.word_pair_similarity_human_scores, m * self.word_pair_similarity_human_scores + b, color="red")

        utils.save_plot(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "data", self.data_directory, "plots", title + ".png"))
        plt.close()


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
            self.context_words = torch.tensor(utils.load_numpy(context_words_filepath), dtype=torch.long, device=device)
            self.target_words = torch.tensor(utils.load_numpy(target_words_filepath), dtype=torch.long, device=device)
            self._num_samples = len(self.target_words)
            progress_bar.update(1)
            return

        context_words = []
        target_words = []
        for sentence in tqdm.tqdm(sentences, desc="Building training data"):
            for center_position, center_word in enumerate(sentence):
                if center_word not in vocabulary or vocabulary.subsample(center_word, threshold=1e-5):
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
        utils.save_numpy(context_words_filepath, context_words)
        utils.save_numpy(target_words_filepath, target_words)

        self.context_words = torch.tensor(context_words, dtype=torch.long, device=device)
        self.target_words = torch.tensor(target_words, dtype=torch.long, device=device)
        self._num_samples = len(self.target_words)
        utils.plot_target_words_occurances(target_words, data_directory="cbow")

    def __iter__(self):
        for start in range(0, self._num_samples, self._batch_size):
            end = min(start + self._batch_size, self._num_samples)
            yield (self.context_words[start:end], self.target_words[start:end])

    def __len__(self):
        return (self._num_samples + self._batch_size - 1) // self._batch_size
