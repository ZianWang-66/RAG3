# University IT Service Desk assistant (RAG over `university_it_policies.txt`)

A retrieval-augmented question-answering assistant built on the policy
knowledge base. It answers questions about university IT guidance, and — most
importantly — it **refuses to answer** when the knowledge base does not contain
the fact, instead of inventing one.

```
Q: What exact phone number should I call for the University IT Service Desk?

A: I don't know the phone number, and I won't guess one. The university IT
   policy documents I have describe contacting the IT Service Desk but they do
   not list any Service Desk telephone number.
```

## Files

| File | Role |
| --- | --- |
| `policy_loader.py` | **Unmodified.** Official loader; the only permitted way to load the corpus. |
| `rag.py` | Chunking, BM25 retrieval, relevance scoring, contact-detail verification. |
| `rag_tool.py` | Retrieval toolkit + LangChain tools + the pinned `qwen3:0.6b` client + the grounding prompt. |
| `agent.py` | Orchestration: retrieve → adjudicate → generate → **validate** → answer. |
| `tests.py` | 48 end-to-end checks, including the target question and the guard rail. |
| `check_data.py` | **Unmodified.** Pre-existing corpus integrity check. |

## Running

The model is served by a local Ollama instance. Nothing is downloaded at run
time, and no model other than `qwen3:0.6b` is ever selected.

```bash
ollama serve                       # if not already running
ollama list                        # expect qwen3:0.6b (the 8B model is not used)

python agent.py                    # interactive
python agent.py "What should I do about a suspicious email?"
python agent.py --verbose "My account is locked"   # shows retrieval diagnostics
python tests.py                    # full test suite
python check_data.py               # corpus integrity
```

> **Interpreter note.** The project `.venv/` in this checkout is a bare
> `virtualenv` skeleton: its `Lib/site-packages` is not writable, so packages
> cannot be installed into it here, and it has no `pip`. The code was therefore
> developed and verified against the interpreter that has the dependencies:
> `C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe`
> (Python 3.13.15). To use the intended venv instead, recreate it and
> `pip install -r requirements.txt`.

## How it works

```
question
   │
   ├─ retrieve (BM25, stemmed tokens, synonym expansion, de-duplicated)
   │
   ├─ adjudicate coverage ──► grounded
   │                          no_contact_detail
   │                          no_match
   │
   ├─ generate with the grounding contract (qwen3:0.6b, think=False)
   │
   ├─ validate the output against the retrieved evidence      ◄── guard rail
   │
   └─ answer, or a canonical refusal
```

### Why retrieval is unavoidable rather than model-chosen

The obvious design is a tool-calling agent: advertise `search_university_it_policies`
and let the model decide when to search. This was measured first, and
`qwen3:0.6b` does not do it — it answers immediately from its own prior
knowledge and emits **no** tool calls. A retrieval-augmented system whose
retrieval is optional is not retrieval augmented. So retrieval runs on every
turn, unconditionally, and the model's job is reduced to what a 0.6B model does
well: phrasing a fact it has been handed. The tools in `rag_tool.py` are still
genuine `langchain_core` tools, but correctness never depends on the model
invoking them.

### Why this question cannot be hallucinated

Three independent layers, in order:

1. **The corpus genuinely lacks the fact.** `check_data.py` states it outright,
   and `tests.py` asserts by regex that no phone number and no email address
   exists anywhere in the 1196 policy documents. The retrieval layer therefore
   has nothing to find.
2. **Coverage adjudication.** `rag.requested_contact_detail()` recognises a
   request for a contact detail, and `rag.verify_evidence()` cross-checks it
   against details *literally extracted* from the retrieved text. A Service
   Desk question with no number in evidence becomes `no_contact_detail`, which
   makes the refusal the required output rather than the model's choice.
3. **The guard rail.** `agent.validate_answer()` re-scans every generated
   answer and rejects any phone number or email address not present in the
   evidence, replacing the whole answer with a refusal. This layer does not
   trust the prompt or the model at all — including deliberately hallucinating
   fake models in `tests.py`, which are caught every time.

Because layer 2 forces the refusal before generation, the target question is
answered correctly *even if the language model is replaced by a worse one*.
Layer 3 is the backstop for every other question.

### Retrieval quality

BM25 over stemmed tokens with IDF-weighted query coverage, scoring in `[0, 1]`.
On-topic questions score `0.67–1.00` against their correct policy topic;
off-topic questions ("How do I bake sourdough bread?", "What is the weather in
Tokyo?") score exactly `0.00` and are rejected. The relevance floor
(`MIN_RELEVANCE = 0.16`, `MIN_BM25 = 1.2` in `rag.py`) is the single place to
tune recall against precision.

The corpus repeats the same guidance across many documents, so retrieval
de-duplicates *textually identical* chunks; otherwise one sentence would fill
the entire evidence window.

## Notes

* `policy_loader.py` and `check_data.py` were not modified. The loader is called
  exactly once, via `documents = load_policy_documents()` in `rag.py`.
* `.venv/` was left untouched; see the interpreter note above.
* The answer to "what can the Service Desk do for me?" style questions is
  grounded in the policy text. The answer to "what is the Service Desk's
  number?" is always, correctly, "I don't know."
