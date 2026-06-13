import string

from nltk import PorterStemmer


class TextProcessor:
    def __init__(self, stopwords: set[str]) -> None:
        self.stemmer = PorterStemmer()
        self.stopwords = {self._normalize(w) for w in stopwords}

    def _normalize(self, text: str) -> str:
        return text.lower().translate(str.maketrans("", "", string.punctuation))

    def preprocess(self, text: str) -> list[str]:
        cleaned = self._normalize(text)
        tokens = [t for t in cleaned.split() if t and t not in self.stopwords]
        return list(map(self.stemmer.stem, tokens))
