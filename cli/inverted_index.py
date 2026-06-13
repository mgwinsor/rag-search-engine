import math
import pickle
from collections import Counter
from pathlib import Path

from nltk import defaultdict

from cli.text_processor import TextProcessor

type Token = str
type DocumentID = int


BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(
        self,
        tokenizer: TextProcessor,
    ) -> None:
        self.index: dict[Token, set[DocumentID]] = defaultdict(set)
        self.docmap: dict[DocumentID, dict] = {}
        self.doc_lengths: dict[DocumentID, int] = {}
        self.term_frequencies: dict[DocumentID, Counter] = defaultdict(Counter)
        self.tokenizer = tokenizer
        self._cache_dir = Path("cache")

    def __add_document(self, doc_id: DocumentID, text: str) -> None:
        tokens = self.tokenizer.preprocess(text)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total = sum([length for length in self.doc_lengths.values()])
        return total / len(self.doc_lengths)

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
            ("doc_lengths.pkl", self.doc_lengths),
        ]:
            with open(self._cache_dir.joinpath(filename), "wb") as f:
                pickle.dump(obj, f)

    def load(self) -> None:
        for filename, attr_name in [
            ("index.pkl", "index"),
            ("docmap.pkl", "docmap"),
            ("term_frequencies.pkl", "term_frequencies"),
            ("doc_lengths.pkl", "doc_lengths"),
        ]:
            with open(self._cache_dir.joinpath(filename), "rb") as f:
                setattr(self, attr_name, pickle.load(f))

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

    def get_bm25_idf(self, term: Token) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: DocumentID, term: Token, k1=BM25_K1, b=BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        doc_normalization = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * doc_normalization)
