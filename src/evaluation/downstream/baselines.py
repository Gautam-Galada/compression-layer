"""Cheap deterministic downstream baselines."""


def identity_context(text: str) -> str:
    return text


def truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    return " ".join(text.split()[:max_tokens])


def extractive_head(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars]
