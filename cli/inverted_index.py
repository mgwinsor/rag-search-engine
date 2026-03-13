from pathlib import Path
import pickle

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
        self.docmap: dict[int, dict] = {}
        self.tokenizer = tokenizer

    def __add_document(self, doc_id: DocumentID, text: str) -> None:
        tokens = self.tokenizer.preprocess(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: Token) -> list[DocumentID]:
        return sorted(self.index.get(term, []))

    def build(self, movies: list[dict]) -> None:
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self) -> None:
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        for filename, obj in [("index.pkl", self.index), ("docmap.pkl", self.docmap)]:
            with open(cache_dir.joinpath(filename), "wb") as f:
                pickle.dump(obj, f)
