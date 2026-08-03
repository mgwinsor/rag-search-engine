import json
import os
from typing import Any, TypedDict

import numpy as np
from click import FileError
from sentence_transformers import SentenceTransformer
from torch.fft import Tensor


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: Tensor | None = None
        self.documents: list[dict] | None = None
        self.document_map: dict[int, dict] = {}

    def generate_embedding(self, text: str | None) -> Tensor:
        if not text or text.isspace():
            raise ValueError("Cannot generate embedding for emtpy text")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> Tensor:
        self.documents = documents
        self.document_map = {}
        movie_strings: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs("cache/", exist_ok=True)
        np.save("cache/movies_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str, Any]]) -> Tensor:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movies_embeddings.npy"):
            self.embeddings = np.load("cache/movies_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                raise FileError("Stale movie embedding cache")
        else:
            return self.build_embeddings(self.documents)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    with open("data/movies.json", "r") as file:
        documents: list[dict] = json.load(file)["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(type(embeddings))
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
