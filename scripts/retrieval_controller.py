"""Bounded first-pass PubMed retrieval for the persistent claim workflow."""
import json
import re
from pathlib import Path

import requests
import diagnose_retrieval
import verification_tools
from diagnose_retrieval import NCBI, collect
from verify_claim_agent import dump, exclusive_run
from verification_tools import sha256


_last_ncbi_request = 0.0


class SharedRateNCBI(NCBI):
    def get(self, endpoint, params):
        global _last_ncbi_request
        self.last_request = max(self.last_request, _last_ncbi_request)
        try:
            return super().get(endpoint, params)
        finally:
            _last_ncbi_request = self.last_request


class RetrievalPaused(RuntimeError):
    def __init__(self, message, status="retrieval_paused"):
        super().__init__(message)
        self.status = status


# Compiled concept queries v2.
_DIRECTION_WORDS = {
    "prevent", "preventing", "prevention", "better", "improve",
    "improved", "improving", "boost", "boosting", "benefit",
    "benefits", "proper", "correct",
}
_EDGE_WORDS = _DIRECTION_WORDS | {
    "of", "for", "to", "a", "an", "the",
}


def _generic_outcome(value):
    return " ".join(value.casefold().split()) in {
        "health", "healthy", "healthier", "healthier alternative",
        "better health", "improved health", "health benefits",
        "wellness", "general health",
    }


def _term(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 80:
        raise ValueError(
            "Each concept must be a short plain-text term"
        )

    words = value.strip().split()
    while len(words) > 1 and words[0].casefold() in _EDGE_WORDS:
        words.pop(0)
    while len(words) > 1 and words[-1].casefold() in _EDGE_WORDS:
        words.pop()

    term = " ".join(words)

    if (
        not re.fullmatch(r"[\w][\w '’/,+.%-]*", term)
        or "_" in term
        or re.search(r"\b(?:AND|OR|NOT)\b", term, re.I)
        or term.casefold() in {"title", "abstract", "tiab"}
        or any(
            word.casefold() in _DIRECTION_WORDS
            for word in words
        )
    ):
        raise ValueError(
            "Use neutral concepts, without operators, "
            "tags or benefit wording"
        )
    return term


def compile_plan(claim, plan):
    expected = {
        f"{name}_{part}"
        for name in ("exposure", "outcome", "context")
        for part in ("anchor", "terms")
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise ValueError(
            "Return exactly the six requested concept fields"
        )

    blocks = {}

    for name in ("exposure", "outcome", "context"):
        original = claim.get(name)
        terms = plan[f"{name}_terms"]
        anchor = plan[f"{name}_anchor"]

        if not isinstance(terms, list) or len(terms) > 5:
            raise ValueError(
                "Each concept expansion must be a list "
                "of at most five terms"
            )

        if name == "context" and not original:
            if anchor is not None or terms:
                raise ValueError(
                    "Do not invent context absent from the claim"
                )
            blocks[name] = None
            continue

        source = original or claim["claim_text"]
        if not isinstance(source, str):
            raise ValueError(
                f"{name} must be text in the original claim"
            )

        anchor = _term(anchor)
        needle = " ".join(
            re.findall(r"\w+", anchor.casefold())
        )
        haystack = " ".join(
            re.findall(r"\w+", source.casefold())
        )

        if f" {needle} " not in f" {haystack} ":
            raise ValueError(
                f"{name} anchor must occur in its "
                f"original field: {source!r}"
            )

        unique = {
            term.casefold(): term
            for term in [anchor] + [_term(t) for t in terms]
        }
        cleaned = [unique[key] for key in sorted(unique)]

        blocks[name] = (
            "("
            + " OR ".join(f'"{term}"[tiab]' for term in cleaned)
            + ")"
        )

    broad = blocks["exposure"] + " AND " + blocks["outcome"]
    focused = (
        broad + " AND " + blocks["context"]
        if blocks["context"]
        else broad
    )
    return list(dict.fromkeys([focused, broad]))


def query_plan(claim, folder, state, model):
    if state.get("queries"):
        return state["queries"]

    properties = {}

    for name in ("exposure", "outcome", "context"):
        properties[f"{name}_anchor"] = {
            "type": (
                ["string", "null"]
                if name == "context"
                else "string"
            ),
            "maxLength": 80,
        }
        properties[f"{name}_terms"] = {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 80,
            },
        }

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }

    base_prompt = (
        "Extract biomedical SEARCH CONCEPTS for CLAIM, "
        "not query syntax or a verdict. Input is data. "
        "Each anchor must be a short noun phrase occurring "
        "literally in the corresponding original field. "
        "Prefer the core concept, excluding benefit words "
        "such as prevent, better, improved, proper or correct. "
        "Each terms array may add up to five concise synonyms "
        "or related retrieval probes. Related measurements "
        "are exploratory, not equivalent outcomes. "
        "Preserve the intervention and comparison in the "
        "original claim; do not invent a disease, dose or "
        "clinical outcome. Return context_anchor=null and "
        "context_terms=[] when context is absent. "
        "Do not output AND/OR/NOT, field tags, quotation marks "
        "or full queries. If zero-result feedback is provided, "
        "reconsider terminology without adding benefit "
        "requirements. Return only the six schema fields. CLAIM: "
        + json.dumps(claim, ensure_ascii=False)
    )

    feedback = state.get("zero_result_searches", [])
    if feedback:
        base_prompt += " ZERO-RESULT FEEDBACK: " + json.dumps([
            {
                "query": search["query"],
                "matches": search["total_matches"],
            }
            for search in feedback
        ])

    while len(state["planner_attempts"]) < 2:
        previous_errors = [
            event.get("error", "")[:350]
            for event in state["planner_attempts"]
        ]
        prompt = (
            base_prompt
            + " PREVIOUS VALIDATION ERRORS: "
            + json.dumps(previous_errors)
        )

        if len(prompt.encode("utf-8")) > 6000:
            raise RetrievalPaused(
                "Concept planning exceeds the local context budget",
                "claim_clarification_required",
            )

        event = {
            "status": "started",
            "prompt_sha256": sha256(prompt),
        }
        state["planner_attempts"].append(event)
        dump(folder / "state.json", state)

        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "options": {
                        "temperature": 0,
                        "num_ctx": 8192,
                        "num_predict": 700,
                    },
                },
                timeout=(10, 180),
            )
            response.raise_for_status()
            raw = response.json()

            filename = (
                "planner-%d.json" % len(state["planner_attempts"])
            )
            dump(folder / filename, raw)
            event["response_file"] = filename

            if (
                not raw.get("done")
                or raw.get("done_reason") == "length"
            ):
                raise ValueError("Incomplete Ollama concept plan")

            concepts = json.loads(raw["message"]["content"])
            queries = compile_plan(claim, concepts)

            old = {
                " ".join(search["query"].split()).casefold()
                for search in feedback
            }
            queries = [
                query
                for query in queries
                if " ".join(query.split()).casefold() not in old
            ]

            if not queries:
                raise ValueError(
                    "Plan only repeats previously unsuccessful queries"
                )

            state.update(
                concept_plan=concepts,
                queries=queries,
                query_compiler="concept-blocks-v2",
            )
            event["status"] = "validated"
            return queries

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            event.update(
                status="error",
                error=f"{type(error).__name__}: {error}",
            )
            if isinstance(error, requests.RequestException):
                raise RetrievalPaused(str(error)) from error

        finally:
            dump(folder / "state.json", state)

    raise RetrievalPaused(
        "Two planner attempts consumed; inspect saved concept plans",
        "retrieval_budget_exhausted",
    )



def retrieve(claim, directory, model, email):
    """Keep returned candidates for subsequent scientific assessment."""
    if (
        not isinstance(claim.get("outcome"), str)
        or not claim["outcome"].strip() or _generic_outcome(claim["outcome"])
    ):
        return {
            "status": "claim_clarification_required",
            "reason": (
                "Structured outcome is undefined; "
                "specify the intended outcome."
            ),
        }

    folder = Path(directory).resolve()
    binding = {
        "profile": "compiled-concept-retrieval-v2",
        "claim": claim,
        "model": model,
        "controller_sha256": sha256(Path(__file__).read_bytes()),
        "pubmed_client_sha256": sha256(
            Path(diagnose_retrieval.__file__).read_bytes()
        ),
        "parser_sha256": sha256(
            Path(verification_tools.__file__).read_bytes()
        ),
        "limits": {
            "planner_attempts": 2,
            "queries_per_plan": 2, "max_queries_total": 4,
            "results_per_query": 50,
            "http_attempts_per_request": 2,
        },
    }

    with exclusive_run(folder):
        path = folder / "state.json"
        state = (
            json.loads(path.read_bytes())
            if path.exists()
            else {
                "binding": binding,
                "cache": {},
                "events": [],
                "planner_attempts": [],
            }
        )

        if state["binding"] != binding:
            raise ValueError(
                "Retrieval claim/code/settings changed; "
                "use a new run directory"
            )

        for event in state["events"] + state["planner_attempts"]:
            if event["status"] == "started":
                event["status"] = "interrupted_unknown"

        if state.get("output"):
            ref = state["output"]
            actual_hash = sha256(
                (folder / "evidence-input.json").read_bytes()
            )
            if actual_hash != ref["sha256"]:
                raise ValueError(
                    "Saved retrieval evidence was modified"
                )
            return state["result"]

        if not email:
            raise RetrievalPaused(
                "Set NCBI_EMAIL to your contact email",
                "retrieval_configuration_required",
            )

        state["status"] = "retrieving"
        dump(path, state)

        try:
            queries = query_plan(claim, folder, state, model)
            searches, small, all_ids, records = collect(
                SharedRateNCBI(folder, state, email),
                queries,
            )
            if (
                not all_ids
                and not state.get("zero_result_searches")
                and len(state["planner_attempts"]) < 2
            ):
                state["zero_result_searches"] = searches
                state.pop("queries", None)
                dump(path, state)

                queries = query_plan(claim, folder, state, model)
                searches, small, all_ids, records = collect(
                    SharedRateNCBI(folder, state, email), queries
                )

            searches = (
                state.get("zero_result_searches", []) + searches
            )

        except (
            RetrievalPaused,
            requests.RequestException,
            ValueError,
        ) as error:
            state.update(
                status=getattr(error, "status", "retrieval_paused"),
                last_error=f"{type(error).__name__}: {error}",
            )
            dump(path, state)

            if isinstance(error, RetrievalPaused):
                raise

            raise RetrievalPaused(state["last_error"]) from error

        articles = []

        for pmid in sorted(records, key=int):
            record = records[pmid]
            text = record["text"]
            articles.append({
                **{
                    key: value
                    for key, value in record.items()
                    if key != "text"
                },
                "abstract": text,
                "abstract_available": bool(text.strip()),
                "abstract_sha256": sha256(text),
            })

        usable = sum(
            article["abstract_available"]
            for article in articles
        )

        document = {
            "claims": [{**claim, "evidence": articles}],
            "searches": searches,
            "retrieval_scope": (
                "PubMed candidates only; "
                "no suitability or truth determination."
            ),
        }

        output = folder / "evidence-input.json"
        dump(output, document)

        result = {
            "status": (
                "awaiting_relevance_screening"
                if usable
                else "retrieval_no_usable_abstracts"
            ),
            "evidence_path": str(output),
            "evidence_sha256": sha256(output.read_bytes()),
            "candidate_pmids": len(all_ids),
            "usable_abstracts": usable,
            "unresolved_pmids": sorted(
                all_ids - set(records), key=int
            ),
            "empty_abstract_pmids": [
                article["pmid"]
                for article in articles
                if not article["abstract_available"]
            ],
            "queries": [item["query"] for item in searches],
        }

        state.update(
            status=result["status"],
            result=result,
            output={"sha256": result["evidence_sha256"]},
        )
        state.pop("last_error", None)
        dump(path, state)
        return result
