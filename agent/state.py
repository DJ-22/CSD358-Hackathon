"""ResearchState: the agent's inspectable research trace (PLAN.md §6). INTERFACE FILE after G-A0."""
from dataclasses import dataclass, field, fields


def _plain(x):
    """Convert sets/tuples/numpy values into JSON-serialisable Python values."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted(_plain(v) for v in x)
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return x.item()  # numpy scalar
    return x


@dataclass
class ResearchState:
    """Agent state across hops: query model, consumed/residual terms, bridges, chains (agentic search trace)."""
    qid: str
    question: str
    qtype: str = "bridge"                                       # "bridge" | "comparison"
    query_terms: dict = field(default_factory=dict)             # stemmed term -> idf
    consumed_terms: set = field(default_factory=set)
    residual_terms: list = field(default_factory=list)          # stemmed
    residual_text: str = ""                                     # unstemmed residual words, question order
    known_entities: set = field(default_factory=set)            # question surfaces + bridges already used
    visited_docs: set = field(default_factory=set)
    hops: list = field(default_factory=list)                    # per hop: query, df-order, candidate count, results
    bridges: list = field(default_factory=list)                 # {entity, source_doc, hand_score, p_e, features}
    chains: list = field(default_factory=list)                  # {d1, entity, d2, s1, p_e, sB, S}
    final_ranking: list = field(default_factory=list)           # [(doc_id, score)]

    def to_json(self) -> dict:
        """Serialise the trace to a JSON-safe dict (sets become sorted lists)."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_json(cls, d: dict) -> "ResearchState":
        """Rebuild a trace from to_json output (lists become sets / tuples again)."""
        s = cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
        s.query_terms = dict(s.query_terms)
        s.consumed_terms = set(s.consumed_terms)
        s.known_entities = set(s.known_entities)
        s.visited_docs = {int(x) for x in s.visited_docs}
        s.final_ranking = [(int(doc), float(score)) for doc, score in s.final_ranking]
        return s
