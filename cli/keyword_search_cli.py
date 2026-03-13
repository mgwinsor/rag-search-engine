#!/usr/bin/env python3

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli.inverted_index import InvertedIndex
from cli.search_movies import search
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
    search_movies = partial(search, preprocessor=text_processor, movies=movies)
    indexer = InvertedIndex(text_processor)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matched_movies = search_movies(args.query)
            for i, movie in enumerate(matched_movies, 1):
                print(f"{i}. {movie['title']}")
        case "build":
            indexer.build(movies)
            indexer.save()
            search_token = text_processor.preprocess("merida")[0]
            search_result = indexer.get_documents(search_token)
            print(search_result[0])
        case _:
            parser.print_help()


def load_movies(path: Path) -> list[dict]:
    data: dict = json.loads(path.read_text())
    return data["movies"]


def load_stopwords(path: Path) -> set[str]:
    return set(path.read_text().splitlines())


if __name__ == "__main__":
    main()
