import pickle
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
        self.tokenizer = tokenizer
        self._cache_dir = Path("cache")

    def __add_document(self, doc_id: DocumentID, text: str) -> None:
        tokens = self.tokenizer.preprocess(text)
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

        for filename, obj in [("index.pkl", self.index), ("docmap.pkl", self.docmap)]:
            with open(self._cache_dir.joinpath(filename), "wb") as f:
                pickle.dump(obj, f)

    def load(self) -> None:
        with open(self._cache_dir.joinpath("index.pkl"), "rb") as f:
            self.index = pickle.load(f)

        with open(self._cache_dir.joinpath("docmap.pkl"), "rb") as f:
            self.docmap = pickle.load(f)
