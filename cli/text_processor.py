import string
from functools import reduce

from nltk import PorterStemmer


class TextProcessor:
    def __init__(self, stopwords: set[str]) -> None:
        self.stemmer = PorterStemmer()
        self.stopwords = stopwords

    def preprocess(self, text: str) -> list[str]:
        transforms = [str.lower, self._remove_punctuation]
        cleaned = reduce(lambda t, f: f(t), transforms, text)
        tokens = [t for t in cleaned.split() if t and t not in self.stopwords]
        return list(map(self.stemmer.stem, tokens))

    def _remove_punctuation(self, text: str):
        return text.translate(str.maketrans("", "", string.punctuation))
