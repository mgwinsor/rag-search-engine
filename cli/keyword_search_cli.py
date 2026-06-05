#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli.inverted_index import InvertedIndex, Token
from cli.text_processor import TextProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    term_frequency = subparsers.add_parser("tf", help="Term Frequency")
    term_frequency.add_argument("doc_id", type=int, help="Document ID")
    term_frequency.add_argument("term", type=str, help="Frequency term")

    idf = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf.add_argument("term", type=str, help="IDF Term")

    args = parser.parse_args()

    movies = load_movies(Path("data/movies.json"))
    stopwords = load_stopwords(Path("data/stopwords.txt"))
    text_processor = TextProcessor(stopwords)
    indexer = InvertedIndex(text_processor)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                indexer.load()
                query_tokens = text_processor.preprocess(args.query)
                results = []
                for token in query_tokens:
                    if len(results) >= 5:
                        break

                    doc_ids = indexer.get_documents(token)
                    for id in doc_ids:
                        if id not in results:
                            results.append(id)

                        if len(results) >= 5:
                            break

                for id in results:
                    print(f"{id}: {indexer.docmap[id]['title']}")
            except FileNotFoundError as e:
                print(f"Error loading files: {e}")
        case "build":
            indexer.build(movies)
            indexer.save()
        case "tf":
            indexer.load()
            term = tokenize_single_term(args.term)
            try:
                freq = indexer.get_tf(args.doc_id, term)
                print(f"Frequency for {term} in document {args.doc_id}: {freq}")
            except KeyError as e:
                print(f"Could not find term: {e}")
                print(0)
        case "idf":
            indexer.load()
            term = tokenize_single_term(args.term)
            total_doc_count = len(indexer.docmap)
            term_match_doc_count = 0
            for doc in indexer.docmap:
                tf = indexer.get_tf(doc, term)
                term_match_doc_count += 1 if tf > 0 else 0
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


def load_movies(path: Path) -> list[dict]:
    data: dict = json.loads(path.read_text())
    return data["movies"]


def load_stopwords(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


def tokenize_single_term(term: str) -> Token:
    stopwords = load_stopwords(Path("data/stopwords.txt"))
    text_processor = TextProcessor(stopwords)
    token = text_processor.preprocess(term)
    if len(token) != 1:
        raise ValueError(f"Expected 1 token, found {len(token)}")
    return token[0]


if __name__ == "__main__":
    main()
