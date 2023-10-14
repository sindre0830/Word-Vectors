import gensim.downloader
import itertools
import nltk
import nltk.corpus
import tqdm


class CorpusLoader():
    def __init__(self) -> None:
        self.corpus = None
        self.is_flattened = False
    
    def build(self, pipeline) -> None:
        for step in tqdm.tqdm(pipeline, desc="Building corpus"):
            step()

    def download(self) -> None:
        self.corpus = gensim.downloader.load("text8")

    def flatten(self) -> None:
        self.corpus = list(itertools.chain.from_iterable(self.corpus))
        self.is_flat = True

    def filter_stop_words(self) -> None:
        nltk.download("stopwords", quiet=True)
        stop_words = nltk.corpus.stopwords.words("english")
        self.corpus = [word for word in self.corpus if word not in stop_words]
