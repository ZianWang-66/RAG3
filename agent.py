"""agent.py — the University IT Service Desk assistant agent.

Orchestration lives here. The pipeline is deliberately *retrieve-first*:

    question
      -> deterministic retrieval + coverage adjudication   (rag.verify_evidence)
      -> route by coverage status:
             grounded          -> model answers from the excerpts only
             no_contact_detail -> model must say the detail is unknown
             no_match          -> model must say it does not know
      -> validate the model's output against the evidence  (guard rail)
      -> return the answer, or a canonical refusal if validation fails

Two design decisions are worth stating plainly, because both are load-bearing
for correctness.

**1. Retrieval does not depend on the model calling a tool.**
The natural design would be a tool-calling agent: advertise
``search_university_it_policies`` and let the model decide when to search. That
was measured first, and qwen3:0.6b does not do it — it answers directly from its
own (unreliable) prior knowledge and emits no tool calls at all. A
retrieval-augmented system whose retrieval is optional is not retrieval
augmented; it is a language model with a suggestion box. So retrieval runs
unconditionally on every turn, and the model's job is reduced to the one thing a
0.6B model does well: extracting and phrasing a fact it has been handed. The
tools in :mod:`rag_tool` are still real ``langchain_core`` tools and are bound
for inspection, but correctness never rests on the model invoking them.

**2. Refusal is enforced by the pipeline, not hoped for from the prompt.**
Prompts are guidance; a 0.6B model can and will drift. So before any answer is
returned, :func:`validate_answer` re-checks it against the retrieved evidence:
any contact detail the model produced that is *not* in the evidence is treated
as an invented fact and the answer is replaced with the canonical refusal. The
prompt asks the model to be honest; the guard rail guarantees it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag import find_contact_details
from rag_tool import (
    ANSWER_DETAIL_MISSING,
    ANSWER_NO_MATCH,
    DEFAULT_SYSTEM_PROMPT,
    MODEL_NAME,
    OllamaChatModel,
    OllamaError,
    RetrievalToolkit,
    build_grounded_prompt,
    detail_refusal,
    get_retrieval_toolkit,
    retrieval_tools,
)

__all__ = [
    "Answer",
    "CANONICAL_REFUSALS",
    "ITServiceDeskAgent",
    "build_agent",
    "finalize_answer",
    "validate_answer",
]


#: Every text the guard rail may substitute for a failed answer. These are the
#: only statements the assistant is allowed to make when it lacks evidence.
CANONICAL_REFUSALS = (ANSWER_NO_MATCH, ANSWER_DETAIL_MISSING)

#: A refusal is recognised by its opening phrase, so harmless trailing wording
#: (a full stop, an offer to help further) does not disqualify a correct
#: refusal. The check is intentionally about *admitting ignorance*, which is the
#: behaviour being protected.
_REFUSAL_OPENERS = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "i cannot find",
    "i can't find",
    "i couldn't find",
    "i could not find",
    "no information",
    "not available in the policy",
    "the policy documents do not",
    "the university it policy documents",
    "that information is not",
    "that is not in the policy",
)


@dataclass
class Answer:
    """A complete, auditable result for one user turn."""

    question: str
    text: str
    status: str
    can_answer: bool
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    guard_triggered: bool = False
    guard_reason: Optional[str] = None
    rewritten_query: Optional[str] = None
    model: str = MODEL_NAME
    raw_model_output: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "status": self.status,
            "can_answer": self.can_answer,
            "guard_triggered": self.guard_triggered,
            "guard_reason": self.guard_reason,
            "rewritten_query": self.rewritten_query,
            "model": self.model,
            "evidence": [
                {
                    "policy_id": row["policy_id"],
                    "topic": row["topic"],
                    "relevance": row["relevance"],
                }
                for row in self.evidence
            ],
        }


# --------------------------------------------------------------------------- #
# The guard rail
# --------------------------------------------------------------------------- #

def _is_refusal(text: str) -> bool:
    """True when ``text`` is an honest "I don't know"-style answer."""
    lowered = (text or "").lower().strip()
    return any(opener in lowered for opener in _REFUSAL_OPENERS)


def validate_answer(
    text: str,
    *,
    status: str,
    evidence_text: str,
) -> Dict[str, Any]:
    """Check a generated answer against the retrieved evidence.

    The hardest failure mode for this task is an invented contact detail, so
    contact details are checked exactly and unconditionally:

    * every phone number in the answer must appear in the evidence;
    * every email address in the answer must appear in the evidence;
    * a question that asked for a missing contact detail must be met with an
      admission of ignorance, not a substitute claim.

    Returns ``{"ok": bool, "reason": str|None, "invented": {...}}``.
    """
    invented: Dict[str, List[str]] = {"phone": [], "email": []}

    answer_details = find_contact_details(text or "")
    evidence_details = find_contact_details(evidence_text or "")

    def normalise(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    evidence_phones = {normalise(p) for p in evidence_details["phone"]}
    evidence_emails = {e.lower() for e in evidence_details["email"]}

    for phone in answer_details["phone"]:
        if normalise(phone) not in evidence_phones:
            invented["phone"].append(phone)

    for email in answer_details["email"]:
        if email.lower() not in evidence_emails:
            invented["email"].append(email)

    if invented["phone"]:
        return {
            "ok": False,
            "reason": (
                "answer contains a phone number that does not appear in the "
                f"retrieved evidence: {invented['phone']}"
            ),
            "invented": invented,
        }

    if invented["email"]:
        return {
            "ok": False,
            "reason": (
                "answer contains an email address that does not appear in the "
                f"retrieved evidence: {invented['email']}"
            ),
            "invented": invented,
        }

    if not (text or "").strip():
        return {
            "ok": False,
            "reason": "model returned an empty answer",
            "invented": invented,
        }

    if status == "no_contact_detail" and not _is_refusal(text):
        # The evidence has no such detail, so an answer that is not an
        # admission of ignorance is an unsupported claim.
        return {
            "ok": False,
            "reason": (
                "question asked for a contact detail that the evidence does not "
                "contain, but the answer neither refused nor cited one"
            ),
            "invented": invented,
        }

    if status == "no_match" and not _is_refusal(text):
        return {
            "ok": False,
            "reason": "no evidence matched, but the answer did not say so",
            "invented": invented,
        }

    return {"ok": True, "reason": None, "invented": invented}


def finalize_answer(
    raw_text: str,
    *,
    status: str,
    evidence_text: str,
    requested_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply the guard rail and return the answer that will actually be shown.

    On a grounded status the model's own wording is kept. On a refusal status
    the canonical sentence is used, because the requirement is that the
    assistant state plainly that it does not know the detail — not that it
    produce a paraphrase of that idea.
    """
    verdict = validate_answer(raw_text, status=status, evidence_text=evidence_text)

    if status == "no_contact_detail":
        return {
            "text": detail_refusal(requested_detail),
            "guard_triggered": not verdict["ok"],
            "guard_reason": verdict["reason"],
            "validated": verdict,
        }

    if status == "no_match":
        return {
            "text": ANSWER_NO_MATCH,
            "guard_triggered": not verdict["ok"],
            "guard_reason": verdict["reason"],
            "validated": verdict,
        }

    if not verdict["ok"]:
        # Grounded status, but the answer invented a contact detail. Replace it
        # with a refusal rather than repairing it: a model that invented a phone
        # number has demonstrated it is not reading the evidence.
        return {
            "text": ANSWER_NO_MATCH,
            "guard_triggered": True,
            "guard_reason": verdict["reason"],
            "validated": verdict,
        }

    return {
        "text": (raw_text or "").strip(),
        "guard_triggered": False,
        "guard_reason": None,
        "validated": verdict,
    }


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class ITServiceDeskAgent:
    """Retrieve-first, evidence-grounded question answering over IT policies."""

    #: Prompt used for the optional keyword-paraphrase search.
    _REWRITE_SYSTEM = (
        "You rewrite a user's question into a short keyword search query for a "
        "university IT policy database. Reply only with JSON."
    )

    _REWRITE_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(
        self,
        llm: Optional[OllamaChatModel] = None,
        toolkit: Optional[RetrievalToolkit] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        rewrite_query: bool = True,
        verbose: bool = False,
        history_turns: int = 4,
    ) -> None:
        self.llm = llm or OllamaChatModel()
        self.toolkit = toolkit or get_retrieval_toolkit()
        self.system_prompt = system_prompt
        #: Search a paraphrase of the question as well as the question itself. A
        #: user asking for a "telephone number" may phrase it as a "helpdesk
        #: hotline"; searching both spellings widens recall without ever
        #: widening what counts as evidence.
        self.rewrite_query = rewrite_query
        self.verbose = verbose
        self.history_turns = history_turns
        self.history: List[Dict[str, str]] = []
        #: Tools bound for introspection and compatibility with models that do
        #: emit tool calls.
        self.tools = list(retrieval_tools)

    # -- logging ----------------------------------------------------------- #

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[agent] {message}", file=sys.stderr)

    # -- query rewriting --------------------------------------------------- #

    def rewrite(self, question: str) -> Optional[str]:
        """Ask the model for a keyword paraphrase of ``question``.

        Best-effort only: any failure simply means no extra search is run. This
        can never *add* evidence that the deterministic retriever would not
        otherwise find, because the paraphrase is fed back through the same BM25
        index and the same relevance floors.
        """
        if not self.rewrite_query:
            return None
        prompt = (
            f"Question: {question}\n\n"
            "Return the most useful search keywords, including any synonyms "
            "(for example phone/telephone/call, or locked/sign-in/password)."
        )
        try:
            raw = self.llm.chat(
                self._REWRITE_SYSTEM, prompt, json_schema=self._REWRITE_SCHEMA
            )
            data = json.loads(raw)
        except (OllamaError, ValueError, TypeError):
            return None
        query = str(data.get("query", "")).strip()
        if not query or query.lower() == question.lower():
            return None
        return query

    # -- evidence gathering ------------------------------------------------ #

    def gather_evidence(self, question: str, rewrite: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve for the question (and its paraphrase) and adjudicate.

        Merging can only make the verdict *more* grounded, and only in one
        direction: ``no_match`` may be upgraded to ``grounded`` when the
        paraphrase finds strong evidence the original wording missed. A
        ``no_contact_detail`` verdict is never upgraded, because that status
        records a fact about the *question* — the user asked for a contact
        detail — and the paraphrase is a different string that would no longer
        contain the request. Upgrading there would silently authorise the model
        to answer a question whose answer the corpus does not hold.
        """
        verdict = self.toolkit.search_and_verify(question)
        if not rewrite:
            return verdict

        extra = self.toolkit.search_and_verify(rewrite)

        seen = {f"{row['policy_id']}#{row['chunk_index']}" for row in verdict["results"]}
        for row in extra["results"]:
            key = f"{row['policy_id']}#{row['chunk_index']}"
            if key not in seen:
                seen.add(key)
                verdict["results"].append(row)

        verdict["strong_result_count"] = max(
            verdict["strong_result_count"], extra["strong_result_count"]
        )
        for kind, values in extra["contact_details_found_in_evidence"].items():
            existing = verdict["contact_details_found_in_evidence"].setdefault(kind, [])
            for value in values:
                if value not in existing:
                    existing.append(value)

        if verdict["status"] == "no_match" and extra["status"] == "grounded":
            verdict["status"] = "grounded"
            verdict["can_answer"] = True
        return verdict

    # -- pipeline ---------------------------------------------------------- #

    def inspect(self, question: str) -> Answer:
        """Run the full pipeline for ``question`` and return an :class:`Answer`."""
        question = (question or "").strip()
        if not question:
            return Answer(
                question="",
                text=ANSWER_NO_MATCH,
                status="no_match",
                can_answer=False,
            )

        rewrite = self.rewrite(question)
        self._log(f"rewritten query: {rewrite!r}")
        verdict = self.gather_evidence(question, rewrite)
        status = verdict["status"]
        self._log(
            f"status={status} strong={verdict['strong_result_count']} "
            f"detail={verdict['requested_detail']!r}"
        )

        evidence_text = self.toolkit.render_evidence(verdict)
        prompt = build_grounded_prompt(question, verdict)

        try:
            raw = self.llm.chat(self.system_prompt, self._with_history(prompt))
        except OllamaError as exc:
            self._log(f"model error: {exc}")
            # Fail closed: without a model there is no validated answer to give,
            # and an unvalidated claim must never be emitted.
            fallback = (
                detail_refusal(verdict.get("requested_detail"))
                if status == "no_contact_detail"
                else ANSWER_NO_MATCH
            )
            return Answer(
                question=question,
                text=fallback,
                status=status,
                can_answer=False,
                evidence=verdict.get("results", []),
                rewritten_query=rewrite,
                raw_model_output=None,
                guard_triggered=True,
                guard_reason=f"model unavailable: {exc}",
            )

        outcome = finalize_answer(
            raw,
            status=status,
            evidence_text=evidence_text,
            requested_detail=verdict.get("requested_detail"),
        )

        answer = Answer(
            question=question,
            text=outcome["text"],
            status=status,
            can_answer=status == "grounded",
            evidence=verdict.get("results", []),
            guard_triggered=outcome["guard_triggered"],
            guard_reason=outcome["guard_reason"],
            rewritten_query=rewrite,
            raw_model_output=raw,
        )
        self._remember(question, answer.text)
        return answer

    def answer(self, question: str) -> str:
        """Convenience wrapper returning just the answer text."""
        return self.inspect(question).text

    # -- conversation ------------------------------------------------------ #

    def _remember(self, question: str, answer: str) -> None:
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.history_turns:
            del self.history[: len(self.history) - self.history_turns]

    def _with_history(self, prompt: str) -> str:
        """Prefix the current turn with a short transcript.

        History is background only: the grounding rules still bind the answer to
        the current EVIDENCE block, and the guard rail validates against that
        block alone, so a previous answer can never become the source of a new
        fact.
        """
        if not self.history:
            return prompt
        transcript = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in self.history
        )
        return (
            "CONVERSATION SO FAR (background only; it is not evidence):\n"
            f"{transcript}\n\n"
            "CURRENT TURN — answer only from the EVIDENCE below.\n\n"
            f"{prompt}"
        )

    def reset(self) -> None:
        """Forget the conversation history."""
        self.history.clear()

    def status(self) -> Dict[str, Any]:
        """Report the model and corpus the agent is running with."""
        return {
            "model": self.llm.model,
            "ollama": self.llm.health(),
            "corpus": self.toolkit.corpus_summary(),
            "tools": [getattr(t, "name", str(t)) for t in self.tools],
        }


def build_agent(verbose: bool = False, **kwargs: Any) -> ITServiceDeskAgent:
    """Construct an :class:`ITServiceDeskAgent` with the default wiring."""
    return ITServiceDeskAgent(verbose=verbose, **kwargs)


# --------------------------------------------------------------------------- #
# Command-line entry point
# --------------------------------------------------------------------------- #

_BANNER = """
University IT Service Desk assistant
  model : {model}
  corpus: {documents} policy documents, {chunks} indexed chunks
  topics: {topics}
Type a question, or 'exit' to quit.
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Interactive loop, or a one-shot answer when questions are arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "--verbose" in argv or "-v" in argv
    words = [a for a in argv if not a.startswith("-")]

    agent = build_agent(verbose=verbose)

    if words:
        result = agent.inspect(" ".join(words))
        print(result.text)
        if verbose:
            print(json.dumps(result.as_dict(), indent=2), file=sys.stderr)
        return 0

    info = agent.status()
    print(
        _BANNER.format(
            model=info["model"],
            documents=info["corpus"]["documents"],
            chunks=info["corpus"]["chunks"],
            topics=", ".join(info["corpus"]["topics"]),
        )
    )
    if not info["ollama"].get("ok"):
        print(f"warning: Ollama model {info['model']!r} not reachable.\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit", ":q"}:
            return 0
        if not question:
            continue
        result = agent.inspect(question)
        print(f"\nassistant> {result.text}\n")
        if verbose:
            print(json.dumps(result.as_dict(), indent=2), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
