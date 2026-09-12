from __future__ import annotations

import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data.evaluation.common import QAPair
from evaluation.metrics import rouge_l_scores, score_answer
from evaluation.summarize_results import _prepare_analysis_view
from pipeline.nodes.common import add_node_metadata
from pipeline.nodes.verifier import _parse_verification


class AnswerScoringTests(unittest.TestCase):
    def test_scores_all_references_and_uses_best_match(self) -> None:
        scores = score_answer(
            "The answer is physical page, paragraph, and line segmentation.",
            [
                "paragraphs and lines",
                "physical page, paragraph, and line segmentation",
            ],
        )

        self.assertEqual(scores["reference_count"], 2)
        self.assertEqual(scores["matched_reference_index"], 1)
        self.assertGreater(scores["f1"], 0.8)

    def test_verbose_correct_answer_exposes_recall_and_verbosity(self) -> None:
        scores = score_answer(
            "Font type and font style are stored. The corpus also contains many "
            "additional layout fields, image coordinates, and page metadata.",
            ["font type and font style"],
        )

        self.assertEqual(scores["answer_recall"], 1.0)
        self.assertEqual(scores["reference_contained"], 1.0)
        self.assertGreater(scores["verbosity_ratio"], 2.0)
        self.assertLess(scores["answer_precision"], scores["answer_recall"])
        self.assertGreater(scores["f2"], scores["f1"])

    def test_rouge_l_rewards_ordered_overlap(self) -> None:
        ordered = rouge_l_scores("alpha beta gamma", "alpha beta gamma")
        reordered = rouge_l_scores("gamma beta alpha", "alpha beta gamma")
        self.assertEqual(ordered["f1"], 1.0)
        self.assertLess(reordered["f1"], ordered["f1"])

    def test_boolean_type_accuracy_accepts_yes_true_equivalence(self) -> None:
        scores = score_answer("Yes, it does.", ["True"], answer_type="yes_no")
        self.assertEqual(scores["type_accuracy"], 1.0)

    def test_official_f1_scores_empty_prediction_zero(self) -> None:
        scores = score_answer("", [""])
        self.assertEqual(scores["f1"], 0.0)

    def test_unanswerable_type_accuracy_accepts_standard_abstention(self) -> None:
        scores = score_answer(
            "Insufficient evidence.", ["unanswerable"], answer_type="unanswerable"
        )
        self.assertEqual(scores["type_accuracy"], 1.0)

    def test_qapair_answers_preserves_legacy_primary_answer(self) -> None:
        pair = QAPair("1", "q", "first", [], ("second", "first"), "free_form")
        self.assertEqual(pair.answers, ["second", "first"])


class VerifierParsingTests(unittest.TestCase):
    def test_parses_separate_verdicts_and_combines_for_routing(self) -> None:
        result = _parse_verification(
            """```json
            {
              "groundedness_verdict": true,
              "relevance_verdict": false,
              "completeness_verdict": true,
              "unsupported_claims": [],
              "reasoning": "Grounded but answers a different question."
            }
            ```"""
        )

        self.assertTrue(result["groundedness_verdict"])
        self.assertFalse(result["relevance_verdict"])
        self.assertFalse(result["verdict"])

    def test_string_false_is_not_treated_as_true(self) -> None:
        result = _parse_verification(
            '{"groundedness_verdict":"false","relevance_verdict":"true",'
            '"completeness_verdict":"true"}'
        )
        self.assertFalse(result["groundedness_verdict"])
        self.assertFalse(result["verdict"])


class MetadataHistoryTests(unittest.TestCase):
    def test_retry_events_are_appended(self) -> None:
        first = add_node_metadata(
            {"retry_count": 0, "metadata": {}},
            "verifier",
            model_id=None,
            input_text="in",
            output_text="out",
            latency_ms=10,
        )
        second = add_node_metadata(
            {"retry_count": 1, "metadata": first},
            "verifier",
            model_id=None,
            input_text="in again",
            output_text="out again",
            latency_ms=20,
        )

        self.assertEqual(len(second["events"]), 2)
        self.assertEqual([event["attempt"] for event in second["events"]], [0, 1])
        self.assertEqual(second["latest"]["verifier"]["latency_ms"], 20)


class RunnerPersistenceTests(unittest.TestCase):
    def test_v2_result_can_be_inserted(self) -> None:
        try:
            import evaluation.runner as runner
        except ModuleNotFoundError as exc:
            self.skipTest(f"full runtime dependencies unavailable: {exc}")

        pair = QAPair(
            "q1",
            "Which fields are stored?",
            "font type and font style",
            ["font type and font style"],
            ("font type and font style", "font information"),
            "extractive",
        )

        class FakeGraph:
            def invoke(self, state: dict) -> dict:
                return {
                    **state,
                    "rewritten_query": "stored typography fields",
                    "retrieved_chunks": ["The corpus stores font type and font style."],
                    "retrieved_metadata": [{"source_id": "paper-1"}],
                    "reranked_chunks": ["The corpus stores font type and font style."],
                    "synthesized_answer": "font type and font style",
                    "verification_result": {
                        "parse_success": True,
                        "verdict": True,
                        "groundedness_verdict": True,
                        "relevance_verdict": True,
                        "completeness_verdict": True,
                        "reasoning": "Directly supported.",
                    },
                    "metadata": {
                        "events": [
                            {
                                "node": "synthesizer",
                                "attempt": 0,
                                "cost_usd": 0.0,
                                "latency_ms": 5,
                            }
                        ],
                        "latest": {},
                    },
                }

        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "results.db"
            with (
                patch.object(runner, "RESULTS_DB_PATH", database_path),
                patch.object(runner, "build_rag_graph", return_value=FakeGraph()),
                patch.object(runner, "_load_pairs", return_value=[pair]),
            ):
                runner.run_evaluation(
                    domain="cs",
                    config_name="homogeneous",
                    model_assignment={
                        "rewriter": "test/model",
                        "retriever": "test/model",
                        "reranker": "test/model",
                        "synthesizer": "test/model",
                        "verifier": "test/model",
                    },
                    n_questions=1,
                    experiment_id="test-experiment",
                    semantic_scoring=False,
                )
                runner.run_evaluation(
                    domain="cs",
                    config_name="homogeneous",
                    model_assignment={
                        "rewriter": "test/model",
                        "retriever": "test/model",
                        "reranker": "test/model",
                        "synthesizer": "test/model",
                        "verifier": "test/model",
                    },
                    n_questions=1,
                    experiment_id="test-experiment",
                    resume=True,
                    semantic_scoring=False,
                )

            import sqlite3

            with closing(sqlite3.connect(database_path)) as connection:
                row = connection.execute(
                    """
                    SELECT reference_count, answer_recall, f2_score, rouge_l_f1,
                           correctness_pass,
                           groundedness_pass, hallucination, verifier_parse_success,
                           evaluation_version, scoring_version, experiment_id
                    FROM results
                    """
                ).fetchone()
            self.assertEqual(
                row,
                (
                    2,
                    1.0,
                    1.0,
                    1.0,
                    1,
                    1,
                    0,
                    1,
                    "2.0-reference-aware",
                    "3.0-official-plus",
                    "test-experiment",
                ),
            )

    def test_summary_view_deduplicates_latest_question(self) -> None:
        import sqlite3

        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute(
                """
                CREATE TABLE results (
                    id INTEGER, domain TEXT, config_name TEXT,
                    model_assignment TEXT, evaluation_version TEXT,
                    scoring_version TEXT, experiment_id TEXT, question_id TEXT
                )
                """
            )
            connection.executemany(
                "INSERT INTO results VALUES (?, 'cs', 'homogeneous', '{}', 'v', 's', NULL, 'q1')",
                [(1,), (2,)],
            )
            _prepare_analysis_view(connection, include_duplicates=False)
            ids = [row[0] for row in connection.execute("SELECT id FROM analysis_results")]
        self.assertEqual(ids, [2])


if __name__ == "__main__":
    unittest.main()
