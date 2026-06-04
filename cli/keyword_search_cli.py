#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli.inverted_index import InvertedIndex
from cli.text_processor import TextProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tokenize_term = subparsers.add_parser("tf", help="Tokenize term")

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
                        if not id in results:
                            results.append(id)

                        if len(results) >= 5:
                            break

                for id in results:
                    print(f"{id}: {indexer.docmap[id]['title']}")
            except FileNotFoundError:
                print("Error loading files...")
        case "build":
            indexer.build(movies)
            indexer.save()
        case "tf":
            pass
        case _:
            parser.print_help()


def load_movies(path: Path) -> list[dict]:
    data: dict = json.loads(path.read_text())
    return data["movies"]


def load_stopwords(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


if __name__ == "__main__":
    main()
