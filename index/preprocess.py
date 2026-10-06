"""Text analysis: tokenisation, case-folding, stop-word removal, Porter stemming."""
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(stopwords.words("english"))
_STEMMER = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(token: str) -> str:
    """Porter stemming (memoised: the vocabulary is far smaller than the token stream)."""
    return _STEMMER.stem(token)


def tokenize(text: str) -> list[str]:
    """Tokenise on alphanumeric runs and case-fold."""
    return [t.lower() for t in _TOKEN.findall(text)]


def analyze_with_positions(text: str, stem: bool = True) -> list[tuple[str, int]]:
    """Analyse text into (term, position) pairs; positions are counted before stop-word removal (positional index)."""
    out = []
    for pos, tok in enumerate(tokenize(text)):
        if tok in _STOP:
            continue
        out.append((_stem(tok) if stem else tok, pos))
    return out


def analyze(text: str, stem: bool = True) -> list[str]:
    """Analyse text into index terms (tokenise, case-fold, stop, stem)."""
    return [t for t, _ in analyze_with_positions(text, stem)]
