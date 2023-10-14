import gensim.downloader
import itertools
import nltk
import nltk.corpus
import tqdm


class CorpusLoader():
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
