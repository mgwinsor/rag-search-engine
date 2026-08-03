import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli.lib.semantic_search import embed_text, verify_embeddings, verify_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded")
    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
