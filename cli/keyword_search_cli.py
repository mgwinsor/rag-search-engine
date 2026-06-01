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
                tokens = text_processor.preprocess(args.query)
                print(tokens)
                results = []
                for token in tokens:
                    results.extend(indexer.get_documents(token))
                for id in results[0:5]:
                    print(f"{id}: {movies[id]['title']}")
            except FileExistsError:
                print("Error loading files...")
        case "build":
            indexer.build(movies)
            indexer.save()
            docs = indexer.get_documents("merida")
            print(f"{docs[0]}")
        case _:
            parser.print_help()


def load_movies(path: Path) -> list[dict]:
    data: dict = json.loads(path.read_text())
    return data["movies"]


def load_stopwords(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


if __name__ == "__main__":
    main()
