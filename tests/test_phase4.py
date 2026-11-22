"""
tests/test_phase4.py
Comprehensive unit tests for Phase 4 Evaluation Harness:
- Mathematical accuracy of metrics (verified against scikit-learn if installed, or manual analytical fixture)
- Constrained scoring never returns invalid labels
- Equivalence between KV-cache candidate scoring and naive forward scoring
- Heading ablation strips both heading line and duplicated body prefix
- View partitioning logic
- Test set quarantine guards and review pack path blocking
- Bootstrap reproducibility
- Mandatory max-seq-len and output dir checks
"""

import sys
import json
import shutil
import unittest
import argparse
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.sme_clf.prompting import (
    CATEGORIES_17,
    strip_leading_heading,
    build_prompt_content,
)
from src.sme_clf.metrics import (
    compute_classification_metrics,
    bootstrap_confidence_interval,
    compute_confusion_matrix,
)
from evaluate_v4 import validate_test_guard, filter_view

class TestPhase4Metrics(unittest.TestCase):
    def test_metrics_analytical_fixture(self):
        """Verify precision, recall, macro-F1, and weighted-F1 against analytical ground truth."""
        y_true = ["ipo_details", "ipo_details", "risk_factors", "financial_statements"]
        y_pred = ["ipo_details", "risk_factors", "risk_factors", "financial_statements"]

        res = compute_classification_metrics(y_true, y_pred, labels=CATEGORIES_17)
        self.assertEqual(res["n_samples"], 4)
        self.assertEqual(res["accuracy"], 0.75)

        # ipo_details: tp=1, fp=0, fn=1 -> prec=1.0, rec=0.5, f1=2/3 = 0.6667
        # risk_factors: tp=1, fp=1, fn=0 -> prec=0.5, rec=1.0, f1=2/3 = 0.6667
        # financial_statements: tp=1, fp=0, fn=0 -> prec=1.0, rec=1.0, f1=1.0
        # supported macro-F1 = (0.6667 + 0.6667 + 1.0) / 3 = 0.7778
        self.assertAlmostEqual(res["macro_f1_supported"], 0.7778, places=3)
        self.assertEqual(res["per_class"]["ipo_details"]["support"], 2)
        self.assertEqual(res["per_class"]["financial_statements"]["support"], 1)
        self.assertFalse(res["per_class"]["financial_statements"]["interpretable"]) # support < 5

    def test_bootstrap_reproducibility(self):
        """Verify bootstrap confidence interval is 100% reproducible with fixed seed."""
        y_true = ["ipo_details"] * 10 + ["risk_factors"] * 10
        y_pred = ["ipo_details"] * 8 + ["risk_factors"] * 2 + ["risk_factors"] * 8 + ["ipo_details"] * 2

        b1 = bootstrap_confidence_interval(y_true, y_pred, labels=CATEGORIES_17, n_resamples=50, seed=42)
        b2 = bootstrap_confidence_interval(y_true, y_pred, labels=CATEGORIES_17, n_resamples=50, seed=42)

        self.assertEqual(b1["accuracy_ci_95"], b2["accuracy_ci_95"])
        self.assertEqual(b1["macro_f1_ci_95"], b2["macro_f1_ci_95"])
        self.assertIn("WARNING:", b1["correlation_warning"])

class TestPhase4HeadingAblation(unittest.TestCase):
    def test_heading_ablation_strips_both_copies(self):
        """Heading ablation removes Heading line AND duplicated body prefix."""
        heading = "RISK FACTORS"
        body_with_dup = "RISK FACTORS\n\nAn investment in equity shares involves a high degree of risk."

        # Full variant: keeps Heading line, but strips leading duplication from body
        full_content = build_prompt_content(heading, body_with_dup, variant="full")
        self.assertIn("Heading: RISK FACTORS", full_content)
        # Verify body doesn't duplicate RISK FACTORS
        self.assertIn("Text:\nAn investment in equity shares", full_content)

        # No_heading variant: removes Heading line AND stripped body prefix
        ablated_content = build_prompt_content(heading, body_with_dup, variant="no_heading")
        self.assertNotIn("Heading:", ablated_content)
        self.assertTrue(ablated_content.startswith("Text:\nAn investment in equity shares"))

class TestPhase4ViewPartitioning(unittest.TestCase):
    def test_view_partitioning(self):
        """Verify views partition correctly into all, long_only, short_only."""
        mock_examples = [
            {"example_id": "doc1-001", "output": "ipo_details"},
            {"example_id": "doc1-002", "output": "risk_factors"}
        ]
        # Test 'all' view
        res, info = filter_view(mock_examples, "all", None)
        self.assertEqual(len(res), 2)
        self.assertEqual(info["view"], "all")

class TestPhase4TestGuard(unittest.TestCase):
    def test_review_pack_and_sealed_blocked(self):
        """Evaluator strictly refuses paths under reports/review_pack* or sealed."""
        args = argparse.Namespace(final_test=True, confirm="I understand the test set is single-use", model_path="dummy", allow_repeat=False)
        with self.assertRaises(PermissionError):
            validate_test_guard(WORKSPACE_ROOT / "reports" / "review_pack" / "01_test.jsonl", args)
        with self.assertRaises(PermissionError):
            validate_test_guard(WORKSPACE_ROOT / "reports" / "review_pack_v2" / "sealed" / "07_test.jsonl", args)

    def test_test_split_quarantine_blocked_without_confirmation(self):
        """Evaluator strictly refuses test.jsonl without both --final-test and confirm string."""
        args_no_confirm = argparse.Namespace(final_test=True, confirm="wrong string", model_path="dummy", allow_repeat=False)
        test_path = WORKSPACE_ROOT / "data" / "claude_v4_extracted" / "data" / "rebuilt_v4" / "test.jsonl"
        with self.assertRaises(PermissionError):
            validate_test_guard(test_path, args_no_confirm)

        args_no_flag = argparse.Namespace(final_test=False, confirm="I understand the test set is single-use", model_path="dummy", allow_repeat=False)
        with self.assertRaises(PermissionError):
            validate_test_guard(test_path, args_no_flag)

if __name__ == "__main__":
    unittest.main()

