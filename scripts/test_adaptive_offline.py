"""Offline integration test for adaptive retrieval."""

import contextlib
import hashlib
import importlib
import io
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from retrieval_state import load_state


class AdaptiveRetrievalTest(unittest.TestCase):
    def test_rounds_and_reassessment(self):
        counters = {"planner": 0, "search": 0}

        # Fake Gemini query generation.
        def generate_queries(**kwargs):
            counters["planner"] += 1
            number = counters["planner"]

            return types.SimpleNamespace(
                output_text=json.dumps([
                    f"stretching AND circulation AND strategy{number}",
                    f"stretching AND bloodflow AND strategy{number}",
                ])
            )

        fake_genai = types.ModuleType("google.genai")
        setattr(
            fake_genai,
            "Client",
            lambda *args, **kwargs: types.SimpleNamespace(
                interactions=types.SimpleNamespace(create=generate_queries)
            ),
        )

        fake_google = types.ModuleType("google")
        setattr(fake_google, "genai", fake_genai)

        # Fake article fetching and ranking.
        fake_retrieval = types.ModuleType("retrieve_evidence")

        def fetch_articles(pmids):
            return [
                {
                    "pmid": str(pmid),
                    "title": f"Offline article {pmid}",
                    "abstract": "Synthetic evidence for testing only.",
                }
                for pmid in pmids
            ]

        def rank_articles(claim, articles):
            return [
                {**article, "relevance_score": 0.5}
                for article in articles
            ]

        setattr(fake_retrieval, "fetch_all_articles", fetch_articles)
        setattr(fake_retrieval, "rerank_articles", rank_articles)
        
        # Each fake search returns one new PMID.
        def search_response(*args, **kwargs):
            counters["search"] += 1
            pmid = str(1000 + counters["search"])

            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "esearchresult": {
                        "count": "1",
                        "idlist": [pmid],
                    }
                },
            )

        def deny_network(*args, **kwargs):
            raise AssertionError("Unexpected real network operation.")

        def write_json(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")

        def run_adaptive(evidence_file, suitability_file):
            arguments = [
                str(SCRIPTS / "adaptive_retrieval.py"),
                str(evidence_file),
                str(suitability_file),
            ]

            with patch.object(sys, "argv", arguments):
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(
                        str(SCRIPTS / "adaptive_retrieval.py"),
                        run_name="__main__",
                    )

        claim = {
            "claim_id": "offline-c001",
            "claim_text": "Stretching after waking improves circulation.",
            "exposure": "stretching",
            "outcome": "circulation",
            "context": "after waking",
        }

        fake_modules = {
            "google": fake_google,
            "google.genai": fake_genai,
            "retrieve_evidence": fake_retrieval,
        }

        previous_directory = Path.cwd()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            evidence_file = root / "evidence.json"
            suitability_file = root / "suitability.json"

            write_json(evidence_file, {
                "video_id": "offline",
                "claims": [{
                    **claim,
                    "evidence": [{
                        "pmid": "1",
                        "title": "Original synthetic article",
                        "abstract": "Offline fixture only.",
                    }],
                }],
            })

            write_json(suitability_file, {
                "claims": [{
                    "claim_id": claim["claim_id"],
                    "needs_more_evidence": True,
                    "missing_evidence": "Evidence gap 1",
                }],
            })

            try:
                import os
                os.chdir(root)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.dict(sys.modules, fake_modules)
                    )
                    stack.enter_context(
                        patch(
                            "requests.sessions.Session.get",
                            side_effect=search_response,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "requests.sessions.Session.request",
                            side_effect=deny_network,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "socket.create_connection",
                            side_effect=deny_network,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "socket.getaddrinfo",
                            side_effect=deny_network,
                        )
                    )

                    reassessment = importlib.import_module(
                        "apply_reassessment"
                    )

                    for number in range(1, 4):
                        run_adaptive(evidence_file, suitability_file)

                        state = load_state(claim)

                        self.assertEqual(
                            state["status"],
                            "awaiting_reassessment",
                        )
                        self.assertEqual(len(state["rounds"]), number)

                        round_record = state["rounds"][-1]
                        input_file = Path(
                            round_record["assessment_input_file"]
                        )

                        input_bytes = input_file.read_bytes()
                        assessment_input = json.loads(input_bytes)
                        articles = assessment_input["claims"][0]["evidence"]

                        # Original article plus two new ones per round.
                        self.assertEqual(len(articles), 1 + 2 * number)

                        pmids = [str(a["pmid"]) for a in articles]
                        self.assertEqual(len(pmids), len(set(pmids)))

                        self.assertEqual(
                            round_record["missing_evidence"],
                            f"Evidence gap {number}",
                        )

                        # No new searches while reassessment is pending.
                        searches_before = counters["search"]

                        with self.assertRaises(SystemExit):
                            run_adaptive(
                                evidence_file,
                                suitability_file,
                            )

                        self.assertEqual(
                            counters["search"],
                            searches_before,
                        )

                        result = {
                            "provenance": {
                                "status": "complete",
                                "input_sha256": hashlib.sha256(
                                    input_bytes
                                ).hexdigest(),
                            },
                            "claims": [{
                                "claim_id": claim["claim_id"],
                                "assessments": [
                                    {
                                        "pmid": pmid,
                                        "exposure_match": True,
                                        "outcome_match": True,
                                        "context_match": False,
                                        "suitability": "partial",
                                        "reason": "Synthetic assessment.",
                                    }
                                    for pmid in pmids
                                ],
                                "needs_more_evidence": True,
                                "missing_evidence": (
                                    f"Evidence gap {number + 1}"
                                ),
                            }],
                        }

                        result_file = root / f"assessment-{number}.json"

                        # Reject an assessment for the wrong input.
                        result["provenance"]["input_sha256"] = "wrong"
                        write_json(result_file, result)

                        before = load_state(claim)

                        with self.assertRaises(ValueError):
                            reassessment.apply_reassessment(
                                input_file,
                                result_file,
                            )

                        self.assertEqual(load_state(claim), before)

                        # Accept the matching assessment.
                        result["provenance"]["input_sha256"] = (
                            hashlib.sha256(input_bytes).hexdigest()
                        )
                        write_json(result_file, result)

                        with contextlib.redirect_stdout(io.StringIO()):
                            reassessment.apply_reassessment(
                                input_file,
                                result_file,
                            )

                        self.assertEqual(
                            load_state(claim)["status"],
                            "needs_planning",
                        )

                    # Round four must stop without searching.
                    searches_before = counters["search"]

                    with self.assertRaises(SystemExit):
                        run_adaptive(evidence_file, suitability_file)

                    state = load_state(claim)

                    self.assertEqual(
                        state["stop_reason"],
                        "max_rounds_reached",
                    )
                    self.assertEqual(len(state["rounds"]), 3)
                    self.assertEqual(
                        counters["search"],
                        searches_before,
                    )

            finally:
                import os
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)