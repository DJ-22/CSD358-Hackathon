"""Dictionary-based entity linker over corpus titles (longest-match n-grams, no stemming)."""
import json
import re
from collections import Counter

from nltk.corpus import stopwords

import config

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_STOP = frozenset(stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    """Tokenise and case-fold without stemming or stop-word removal (surface-form matching)."""
    return [t.lower() for t in _TOKEN.findall(text)]


def title_surface(title: str) -> str:
    """Title surface form: lowercase, trailing parenthetical disambiguator stripped, unstemmed tokens joined by spaces."""
    return " ".join(tokenize(_TRAILING_PAREN.sub("", title)))


class EntityLinker:
    """Longest-match n-gram entity linker mapping title surfaces to doc_ids."""

    def __init__(self, corpus: str):
        """Build the title-surface dictionary and token document frequencies from the corpus files."""
        self.corpus = corpus
        title2doc = json.loads((config.DATA_PROCESSED / f"title2doc_{corpus}.json").read_text(encoding="utf-8"))
        self.surface2docs: dict[tuple[str, ...], list[int]] = {}
        self.doc2surface: dict[int, str] = {}
        for title, doc_id in title2doc.items():
            surface = title_surface(title)
            self.doc2surface[int(doc_id)] = surface
            if surface:
                self.surface2docs.setdefault(tuple(surface.split()), []).append(int(doc_id))
        for docs in self.surface2docs.values():
            docs.sort()
        self.df: Counter = Counter()
        with open(config.DATA_PROCESSED / f"corpus_{corpus}.jsonl", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                self.df.update(set(tokenize(doc["title"] + " " + doc["text"])))

    def _linkable(self, tokens: tuple[str, ...]) -> bool:
        """Filter non-entity surfaces: pure numbers, single stop words and single high-df (non-discriminative) tokens."""
        if all(t.isdigit() for t in tokens):
            return False
        if len(tokens) == 1:
            t = tokens[0]
            return t not in _STOP and self.df[t] <= config.MAX_ENTITY_DF
        return True

    def link(self, text: str) -> list[tuple[str, list[int], int]]:
        """Entity linking: (surface, doc_ids, start) via longest-match sliding n-grams, no overlaps."""
        tokens = tokenize(text)
        matches = []
        for i in range(len(tokens)):
            for n in range(min(config.MAX_NGRAM, len(tokens) - i), 0, -1):
                gram = tuple(tokens[i:i + n])
                if gram in self.surface2docs and self._linkable(gram):
                    matches.append((n, i))
                    break  # longest match starting at i
        matches.sort(key=lambda m: (-m[0], m[1]))  # longest first, then leftmost
        taken = [False] * len(tokens)
        out = []
        for n, i in matches:
            if any(taken[i:i + n]):
                continue
            taken[i:i + n] = [True] * n
            gram = tuple(tokens[i:i + n])
            out.append((" ".join(gram), list(self.surface2docs[gram]), i))
        out.sort(key=lambda e: e[2])
        return out
