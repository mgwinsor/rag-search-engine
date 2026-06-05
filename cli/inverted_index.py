import math
import pickle
from collections import Counter
from pathlib import Path

from nltk import defaultdict

from cli.text_processor import TextProcessor

type Token = str
type DocumentID = int


class InvertedIndex:
    def __init__(
        self,
        tokenizer: TextProcessor,
    ) -> None:
        self.index: dict[Token, set[DocumentID]] = defaultdict(set)
        self.docmap: dict[DocumentID, dict] = {}
        self.term_frequencies: dict[DocumentID, Counter] = defaultdict(Counter)
        self.tokenizer = tokenizer
        self._cache_dir = Path("cache")

    def __add_document(self, doc_id: DocumentID, text: str) -> None:
        tokens = self.tokenizer.preprocess(text)
        self.term_frequencies[doc_id].update(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: Token) -> list[DocumentID]:
        return sorted(self.index.get(term, []))

    def build(self, movies: list[dict]) -> None:
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        for filename, obj in [
            ("index.pkl", self.index),
            ("docmap.pkl", self.docmap),
            ("term_frequencies.pkl", self.term_frequencies),
        ]:
            with open(self._cache_dir.joinpath(filename), "wb") as f:
                pickle.dump(obj, f)

    def load(self) -> None:
        with open(self._cache_dir.joinpath("index.pkl"), "rb") as f:
            self.index = pickle.load(f)

        with open(self._cache_dir.joinpath("docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)

        with open(self._cache_dir.joinpath("term_frequencies.pkl"), "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: DocumentID, term: Token) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: Token) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: DocumentID, term: Token) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
