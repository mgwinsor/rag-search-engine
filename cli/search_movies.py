import itertools

from cli.text_processor import TextProcessor


def search(
    query: str,
    preprocessor: TextProcessor,
    movies: list[dict],
    result_limit: int = 5,
) -> list[dict]:
    query_tokens = preprocessor.preprocess(query)
    matches = (m for m in movies if _is_match(preprocessor, m["title"], query_tokens))
    return list(itertools.islice(matches, result_limit))


def _is_match(preprocessor: TextProcessor, title: str, query_tokens: list[str]) -> bool:
    title_tokens = preprocessor.preprocess(title)
    return any(qt in tt for qt in query_tokens for tt in title_tokens)
