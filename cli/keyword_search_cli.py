#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli.inverted_index import BM25_K1, InvertedIndex, Token
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

    tfidf = subparsers.add_parser("tfidf", help="Calculate TF-IDF")
    tfidf.add_argument("doc_id", type=int, help="Document ID")
    tfidf.add_argument("term", type=str, help="Term to calculate TF-IDF")

    bm25idf = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25idf.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25tf = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given term")
    bm25tf.add_argument("doc_id", type=int, help="Document ID")
    bm25tf.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25tf.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

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
            tf = indexer.get_tf(args.doc_id, tokenize_single_term(args.term))
            print(f"Frequency for {args.term} in document {args.doc_id}: {tf}")
        case "idf":
            indexer.load()
            idf = indexer.get_idf(tokenize_single_term(args.term))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            indexer.load()
            tf_idf = indexer.get_tf_idf(args.doc_id, tokenize_single_term(args.term))
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            indexer.load()
            bm25idf = indexer.get_bm25_idf(tokenize_single_term(args.term))
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            indexer.load()
            k1 = BM25_K1
            if args.k1:
                k1 = args.k1
            bm25tf = indexer.get_bm25_tf(
                args.doc_id, tokenize_single_term(args.term), k1
            )
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
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
