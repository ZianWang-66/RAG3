"""rag.py — retrieval layer for the University IT Service Desk assistant.

This module builds the *knowledge* half of the system:

    load_policy_documents()  ->  chunking  ->  BM25 index  ->  retrieve()

and the *verification* half:

    relevance scoring        ->  contact-detail extraction  ->  evidence verdicts

The retrieval design is deliberately lexical (BM25 over stemmed tokens) rather
than embedding based. Two reasons:

1. The knowledge base is a small set of dense university IT guidance
   documents. For a corpus this small and this topically repetitive, lexical
   overlap plus IDF weighting discriminates between policies at least as well
   as a small embedding model, and it is exactly reproducible.
2. No embedding model is available locally, and no source of truth about
   "contacting the Service Desk by phone" exists in the corpus. A fuzzy
   semantic retriever would happily return a *vaguely related* chunk for a
   question the corpus cannot answer; BM25 with an explicit relevance floor
   lets us say so honestly.

Everything in this file is pure standard library plus ``langchain_core`` (used
only for the ``Document`` container that ``policy_loader`` already returns).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# The ONLY permitted way to load the document. Do not change this call.
from policy_loader import load_policy_documents


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: How many retrieved chunks are handed to the generator as evidence.
DEFAULT_TOP_K: int = 4

#: BM25 term-frequency saturation and length-normalisation parameters. The
#: standard Robertson/Sparck-Jones defaults.
BM25_K1: float = 1.5
BM25_B: float = 0.75

#: Minimum normalised relevance a chunk must reach to count as evidence.
#: Scores are cosine-like in [0, 1] (see :meth:`PolicyIndex._coverage`), so this
#: is a meaningful absolute floor rather than a moving target.
MIN_RELEVANCE: float = 0.16

#: BM25 scores are unbounded, so the "strong match" test uses a floor that was
#: calibrated against the real corpus (see tests/). Anything below this is noise.
MIN_BM25: float = 1.2

#: Number of sentences grouped into one chunk, and the sentence overlap.
CHUNK_SENTENCES: int = 3
CHUNK_OVERLAP: int = 1


STOPWORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
    # Words that occur in almost every policy document and therefore carry no
    # discriminating power for this corpus.
    "user", "users", "university", "policy", "policies", "guidance",
    "applies", "routine", "support", "request", "requests", "student",
    "students", "may", "also", "must", "exact",
}

#: Query-side vocabulary expansion. Written as phrases so multi-word keys are
#: matched before single tokens. This only ever *adds* search vocabulary; it can
#: never introduce an answer, and every expansion term still has to exist in the
#: corpus for the corresponding chunk to score.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "phone number": ("phone", "telephone", "number", "call", "desk"),
    "email address": ("email", "mailbox"),
    "lock out": ("locked", "account", "password"),
    "sign in": ("signin", "login", "account", "password"),
    "log in": ("signin", "login", "account", "password"),
    "wi fi": ("wifi", "wireless", "network"),
    "remote access": ("vpn", "network", "remote"),
    "disk space": ("storage", "quota", "drive"),
    "phone": ("call", "telephone", "contact", "desk"),
    "telephone": ("phone", "call", "desk"),
    "hotline": ("phone", "call", "desk"),
    "helpline": ("phone", "call", "desk"),
    "extension": ("phone", "call", "desk"),
    "dial": ("call", "phone", "desk"),
    "contact": ("desk", "email", "call"),
    "locked": ("lock", "signin", "password", "account"),
    "signin": ("login", "account", "password"),
    "wifi": ("wireless", "network"),
    "internet": ("wifi", "network", "vpn"),
    "vpn": ("network", "remote"),
    "print": ("printer", "printing", "queue", "credit"),
    "printer": ("print", "printing", "queue"),
    "printing": ("print", "printer", "queue", "credit"),
    "phishing": ("suspicious", "email", "report"),
    "spam": ("phishing", "suspicious", "email"),
    "scam": ("phishing", "suspicious", "email"),
    "storage": ("quota", "drive", "space"),
    "password": ("account", "reset", "portal"),
    "reset": ("password", "account", "portal"),
    "account": ("password", "signin", "locked"),
    "mail": ("email", "mailbox", "phishing"),
    "mailbox": ("email", "quota"),
    "deleted": ("deletion", "hour", "retention"),
    "kept": ("retention", "deleted", "stored"),
}

#: Irregular/plural forms so tokens match across sentences. Applied before a
#: conservative suffix-stripping pass.
LEMMA_OVERRIDES: Dict[str, str] = {
    "passwords": "password", "accounts": "account", "printers": "printer",
    "emails": "email", "messages": "message", "attempts": "attempt",
    "settings": "setting", "hours": "hour", "minutes": "minute",
    "days": "day", "jobs": "job", "devices": "device", "services": "service",
    "viruses": "virus", "browsers": "browser", "files": "file",
    "drives": "drive", "quotas": "quota", "cards": "card", "campuses": "campus",
}


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _stem(token: str) -> str:
    """Crudely normalise an English token.

    Only inflectional suffixes are removed, and never from short tokens, so
    ``printing`` and ``print`` collapse while ``storage`` and ``store`` do not.
    """
    if token in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[token]
    if len(token) > 4:
        for suffix in ("ing", "ers", "er", "ies", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                base = token[: -len(suffix)]
                if suffix == "ies":
                    base = base[:-1] + "y"
                return base
    return token


def tokenize(text: str) -> List[str]:
    """Lower-case, split, drop stopwords and normalise inflections."""
    return [
        _stem(tok)
        for tok in _WORD_RE.findall(text.lower())
        if tok not in STOPWORDS and len(tok) > 1
    ]


def _phrase_variants(text: str) -> str:
    """Append synonym vocabulary to ``text`` (never replace it)."""
    lowered = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    padded = f" {lowered} "
    additions: List[str] = []
    for key in sorted(SYNONYMS, key=len, reverse=True):
        if f" {key} " in padded:
            additions.extend(SYNONYMS[key])
    if additions:
        return text + " " + " ".join(additions)
    return text


def _split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of one policy document."""

    text: str
    policy_id: str
    topic: str
    source: str
    chunk_index: int

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "topic": self.topic,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus the scores that justified retrieving it."""

    chunk: Chunk
    bm25: float
    coverage: float
    matched_terms: Tuple[str, ...]

    @property
    def relevance(self) -> float:
        """Normalised relevance in [0, 1] used for every downstream decision."""
        return self.coverage

    def as_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.chunk.policy_id,
            "topic": self.chunk.topic,
            "chunk_index": self.chunk.chunk_index,
            "relevance": round(self.relevance, 4),
            "bm25": round(self.bm25, 4),
            "matched_terms": list(self.matched_terms),
            "text": self.chunk.text,
        }


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_documents(documents: Sequence[Any]) -> List[Chunk]:
    """Split policy documents into overlapping sentence windows.

    Every chunk inherits ``policy_id``/``topic``/``source`` metadata from the
    parent :class:`~langchain_core.documents.Document`, which is what makes a
    citation back to a specific policy possible.
    """
    chunks: List[Chunk] = []
    for doc in documents:
        text = (getattr(doc, "page_content", "") or "").strip()
        if not text:
            continue
        meta = getattr(doc, "metadata", {}) or {}
        sentences = _split_sentences(text)

        step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
        windows: List[str] = []
        for start in range(0, len(sentences), step):
            window = sentences[start : start + CHUNK_SENTENCES]
            if not window:
                break
            windows.append(" ".join(window))
            if start + CHUNK_SENTENCES >= len(sentences):
                break

        # A whole-policy chunk is kept alongside the windows: some questions
        # ("what does the printing policy cover?") are best served by the full
        # text, and it guarantees no sentence is unreachable.
        if len(windows) > 1:
            windows.append(text)

        for index, window in enumerate(windows):
            chunks.append(
                Chunk(
                    text=window,
                    policy_id=str(meta.get("policy_id", "unknown")),
                    topic=str(meta.get("topic", "unknown")),
                    source=str(meta.get("source", "unknown")),
                    chunk_index=index,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

class PolicyIndex:
    """A BM25 index over :class:`Chunk` objects.

    Built once at import time from the policy documents. Building is O(n) in
    corpus size and querying is O(n) in chunk count, which keeps the whole
    system free of native dependencies and exactly reproducible.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = BM25_K1, b: float = BM25_B):
        self.chunks: List[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b
        # Term frequencies and lengths are all that BM25 needs; the raw token
        # lists are not retained.
        counts: List[Counter] = [Counter(tokenize(c.text)) for c in self.chunks]
        self._counts: List[Counter] = counts
        self._lengths: List[int] = [sum(c.values()) for c in counts]

        n_docs = len(self.chunks)
        self.avg_length: float = sum(self._lengths) / n_docs if n_docs else 0.0

        doc_freq: Counter = Counter()
        for counts in self._counts:
            doc_freq.update(counts.keys())
        # Smoothed IDF: strictly positive, so no term can contribute a negative
        # score and every matched term is evidence of topical overlap.
        self.idf: Dict[str, float] = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    # -- scoring ----------------------------------------------------------- #

    def _bm25(self, query_terms: Iterable[str], doc_index: int) -> Tuple[float, List[str]]:
        counts = self._counts[doc_index]
        length = self._lengths[doc_index] or 1
        norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
        score = 0.0
        matched: List[str] = []
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            matched.append(term)
            idf = self.idf.get(term, 0.0)
            score += idf * (freq * (self.k1 + 1)) / (freq + norm)
        return score, matched

    def _coverage(self, query_terms: Sequence[str], doc_index: int) -> float:
        """IDF-weighted share of the query's discriminative mass present here.

        This is the score used for the relevance floor, because unlike raw BM25
        it is comparable across queries of different lengths and topics.
        """
        unique = list(dict.fromkeys(query_terms))
        if not unique:
            return 0.0
        total = sum(self.idf.get(t, 0.0) for t in unique)
        if total <= 0:
            return 0.0
        counts = self._counts[doc_index]
        hit = sum(self.idf.get(t, 0.0) for t in unique if counts.get(t, 0))
        return hit / total

    # -- querying ---------------------------------------------------------- #

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        prefer_unique: bool = True,
    ) -> List[RetrievedChunk]:
        """Return the ``top_k`` best chunks for ``query``, best first.

        ``prefer_unique`` collapses *textually identical* chunks (the corpus
        restates the same guidance across many documents). Without it the whole
        evidence window can be filled with copies of one sentence; with it the
        generator sees distinct guidance, which is what makes a grounded answer
        possible. It never hides a match, because the first collision is always
        kept and only exact restatements are dropped.
        """
        terms = tokenize(_phrase_variants(query))
        if not terms:
            return []

        scored: List[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            bm25, matched = self._bm25(terms, index)
            if bm25 <= 0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    bm25=bm25,
                    coverage=self._coverage(terms, index),
                    matched_terms=tuple(sorted(set(matched))),
                )
            )

        # Rank by BM25, then relevance, then a stable policy/chunk order so the
        # output is deterministic run to run.
        scored.sort(
            key=lambda r: (-r.bm25, -r.coverage, r.chunk.policy_id, r.chunk.chunk_index)
        )

        if not prefer_unique:
            return scored[:top_k]

        seen: set = set()
        unique: List[RetrievedChunk] = []
        for item in scored:
            fingerprint = re.sub(r"[\W_]+", " ", item.chunk.text.lower()).strip()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(item)
            if len(unique) >= top_k:
                break
        return unique


# --------------------------------------------------------------------------- #
# Build the index from the (unmodifiable) loader
# --------------------------------------------------------------------------- #

documents = load_policy_documents()

documents_by_id: Dict[str, Any] = {
    str(getattr(doc, "metadata", {}).get("policy_id")): doc for doc in documents
}

CHUNKS: List[Chunk] = chunk_documents(documents)

INDEX = PolicyIndex(CHUNKS)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    prefer_unique: bool = True,
) -> List[RetrievedChunk]:
    """Convenience wrapper around :meth:`PolicyIndex.search`."""
    return INDEX.search(query, top_k=top_k, prefer_unique=prefer_unique)


# --------------------------------------------------------------------------- #
# Contact-detail verification
# --------------------------------------------------------------------------- #
#
# The knowledge base never states a Service Desk telephone number. Nothing in
# this module invents one: these helpers *extract and report* what is literally
# present in the retrieved text, so the agent can distinguish
#
#     "the corpus describes a Service Desk but gives no number"
# from
#     "the corpus gives a number"   (which it never does for this dataset)
#
# and so that any contact detail appearing in a generated answer can be checked
# against the evidence before the answer is shown to the user.

#: Phone shapes: international, local, parenthesised, hyphenated, spaced, or a
#: plain run of digits. Deliberately permissive, because the point is to detect
#: *any* number so that an invented one can never slip past verification.
PHONE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\+\d[\d\s().\-]{5,}\d"),
    re.compile(r"\(\d{2,5}\)\s*\d[\d\s\-]{4,}\d"),
    re.compile(r"\b(?:1[ -]?800|1[ -]?888|800|888)[ -]?\d{3}[ -]?\d{4}\b"),
    re.compile(r"\b\d{3,4}[\s.\-]\d{3,4}(?:[\s.\-]\d{2,4})?\b"),
    re.compile(r"\b\d{5,}\b"),
    re.compile(r"\b(?:ext|extension)\.?\s*[:#]?\s*\d{1,6}\b", re.IGNORECASE),
)


def find_contact_details(text: str) -> Dict[str, List[str]]:
    """Extract the contact details literally present in ``text``."""
    phones: List[str] = []
    for pattern in PHONE_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = match.group(0).strip()
            if value and value not in phones:
                phones.append(value)
    emails = list(dict.fromkeys(re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text or "")))
    urls = list(
        dict.fromkeys(re.findall(r"https?://\S+|www\.\S+", text or "", flags=re.IGNORECASE))
    )
    return {"phone": phones, "email": emails, "url": urls}


#: Phrases that indicate the user is *requesting* a contact detail, as opposed
#: to merely mentioning a topic that happens to contain one of these words.
#: Deliberately narrow: "a suspicious email" is a topic, whereas "the email
#: address" is a request. Requiring the qualified phrase here is what stops the
#: miss-classification that would otherwise turn an answerable question into a
#: spurious refusal. Invention is handled separately by the guard rail in
#: :mod:`agent`, which validates the generated answer against the evidence.
_DETAIL_TERMS: Dict[str, Tuple[str, ...]] = {
    "phone": (
        "phone number", "telephone number", "contact number", "phone no",
        "phone", "telephone", "hotline", "helpline", "helpdesk number",
        "service desk number", "desk number", "extension number", "extension",
    ),
    "email": ("email address", "e-mail address", "mail address", "mailbox address"),
    "url": ("url", "website", "web address", "portal address", "web page address"),
}

#: Asking to *call/dial/ring* something, or asking for "the number", is a
#: phone-number request even when the verb and the noun are separated
#: ("How do I call the Service Desk? What number?"). Detecting these up front
#: matters: it lets the pipeline *force* a refusal instead of relying on the
#: model to choose one. The window is generous because a user's request is
#: often split across two sentences, while the verbs involved ("dial", "ring",
#: "hotline", "telephone number") have no unrelated meaning in this domain.
_PHONE_INTENT_RE = re.compile(
    r"(?:\b(?:call|dial|ring|phone|telephone|hotline|helpline)\b.{0,60}?"
    r"\b(?:number|num|no|ext|extension)\b)"
    r"|(?:\b(?:number|num|ext|extension)\b.{0,60}?"
    r"\b(?:call|dial|ring)\b)",
    re.IGNORECASE | re.DOTALL,
)


def requested_contact_detail(question: str) -> Optional[str]:
    """Return ``"phone"``/``"email"``/``"url"`` if the question *asks for* one.

    Used as an extra guard: if a user asks for a contact detail of a kind that
    the retrieved evidence does not contain, the honest answer is "I don't have
    that", regardless of how relevant the retrieved policy text looks. Because
    this check is what triggers a forced refusal, it must also cover indirect
    phrasings such as "what number do I dial?".
    """
    lowered = f" {re.sub(r'[^a-z0-9]+', ' ', (question or '').lower())} "
    for kind in ("phone", "email", "url"):
        for term in _DETAIL_TERMS[kind]:
            spaced = term.replace("-", " ")
            if f" {spaced} " in lowered or f" {spaced}s " in lowered:
                return kind
    if _PHONE_INTENT_RE.search(question or ""):
        return "phone"
    return None


def _strong(item: RetrievedChunk) -> bool:
    return item.relevance >= MIN_RELEVANCE and item.bm25 >= MIN_BM25


def verify_evidence(question: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """Retrieve for ``question`` and adjudicate whether the corpus can answer.

    Returns a structured verdict with three possible outcomes:

    ``grounded``
        At least one chunk is a strong, on-topic match. The generator may
        answer, but only from the supplied excerpts.
    ``no_contact_detail``
        The question explicitly *asks for* a phone number / email address / URL
        and the retrieved policies describe the topic but never state that
        detail. The generator must say the detail is not available.
    ``no_match``
        Nothing in the corpus is relevant enough. The generator must say it
        does not know.
    """
    detail_kind = requested_contact_detail(question)
    results = retrieve(question, top_k=top_k, prefer_unique=True)
    strong = [item for item in results if _strong(item)]

    evidence_text = "\n".join(item.chunk.text for item in results)
    found = find_contact_details(evidence_text)

    if not strong:
        status = "no_match"
    elif detail_kind and not found.get(detail_kind):
        status = "no_contact_detail"
    else:
        status = "grounded"

    return {
        "status": status,
        "question": question,
        "requested_detail": detail_kind,
        "can_answer": status == "grounded",
        "results": [item.as_dict() for item in results],
        "strong_result_count": len(strong),
        "contact_details_found_in_evidence": found,
    }


def format_evidence(results: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, citable evidence block."""
    if not results:
        return "(no excerpts retrieved)"
    blocks: List[str] = []
    for position, item in enumerate(results, start=1):
        blocks.append(
            f"[{position}] source={item.chunk.source} "
            f"policy_id={item.chunk.policy_id} topic={item.chunk.topic} "
            f"relevance={item.relevance:.2f}\n{item.chunk.text}"
        )
    return "\n\n".join(blocks)


__all__ = [
    "CHUNKS",
    "DEFAULT_TOP_K",
    "INDEX",
    "MIN_BM25",
    "MIN_RELEVANCE",
    "Chunk",
    "PolicyIndex",
    "RetrievedChunk",
    "chunk_documents",
    "documents",
    "documents_by_id",
    "find_contact_details",
    "format_evidence",
    "requested_contact_detail",
    "retrieve",
    "tokenize",
    "verify_evidence",
]
