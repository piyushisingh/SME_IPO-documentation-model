"""
tests/test_phase3.py
Comprehensive unit tests for Phase 3 components using standard library unittest.
"""

import os
import sys
import json
import shutil
import unittest
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from transformers import AutoTokenizer

from src.sme_clf.prompting import (
    CATEGORIES_17,
    SHORT_INSTRUCTION,
    PROMPT_VERSION,
    PROMPT_HASH,
    normalize_for_match,
    strip_leading_heading,
    build_prompt_content,
    build_messages,
    build_full_ids,
    build_prompt_ids,
)
from src.sme_clf.build_dataset import check_eligibility_every_source
from train_v4 import validate_safety

DEFAULT_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--unsloth--gemma-3-1b-it-unsloth-bnb-4bit" / "snapshots" /
    "bfae90ad90f41a9be81bd68255d733bb80c97159"
)

class TestPhase3Prompting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(DEFAULT_SNAPSHOT),
            local_files_only=True
        )

    def test_prompt_format_and_single_bos(self):
        """Instruction text matches short form and produces valid tokens with exactly 1 BOS."""
        self.assertEqual(
            SHORT_INSTRUCTION,
            "Classify the following SME IPO prospectus section into the most appropriate document category."
        )
        instr_tokens = self.tokenizer.encode(SHORT_INSTRUCTION, add_special_tokens=False)
        self.assertEqual(len(instr_tokens), 15)

        full_ids = build_full_ids(
            heading="FINANCIAL STATEMENTS",
            body="The financial summary shows revenue growth.",
            target="financial_statements",
            tokenizer=self.tokenizer
        )
        # Verify exactly one BOS token at index 0
        self.assertEqual(full_ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(full_ids.count(self.tokenizer.bos_token_id), 1)

    def test_heading_ablation(self):
        """strip_leading_heading properly removes matching heading, leaves non-matching alone."""
        h1 = "DEFINITIONS AND ABBREVIATIONS"
        body_with_h1 = "DEFINITIONS AND ABBREVIATIONS\n\nThis Draft Red Herring Prospectus uses certain definitions..."
        stripped = strip_leading_heading(body_with_h1, h1)
        self.assertTrue(stripped.startswith("This Draft Red Herring Prospectus"))

        # Non-matching
        h2 = "RISK FACTORS"
        body2 = "Summary of the issue:\n\nThe issue comprises..."
        stripped2 = strip_leading_heading(body2, h2)
        self.assertEqual(stripped2, body2)

    def test_target_preservation_under_extreme_length(self):
        """Input with 20,000 characters truncates body but strictly preserves target tokens."""
        long_body = "word " * 10000 # ~50,000 chars, ~10,000 tokens
        target_cat = "risk_factors"
        max_seq = 1024
        heading = "LONG SECTION"

        # Emulate token-aware truncation from build_dataset.py
        dummy_full_ids = build_full_ids(heading=heading, body="", target=target_cat, tokenizer=self.tokenizer)
        fixed_tokens = len(dummy_full_ids)
        available_body_tokens = max(16, max_seq - fixed_tokens - 4)

        body_tokens_full = self.tokenizer(long_body, add_special_tokens=False)["input_ids"]
        body_tokens_kept = body_tokens_full[:available_body_tokens]
        body_kept_text = self.tokenizer.decode(body_tokens_kept, skip_special_tokens=True).rstrip()

        full_ids = build_full_ids(
            heading=heading,
            body=body_kept_text,
            target=target_cat,
            tokenizer=self.tokenizer
        )

        self.assertLessEqual(len(full_ids), max_seq)
        self.assertEqual(full_ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(full_ids.count(self.tokenizer.bos_token_id), 1)

        # Check that the ending tokens decode to the target category
        rendered = self.tokenizer.decode(full_ids)
        self.assertTrue(rendered.strip().endswith(target_cat + "<end_of_turn>"))

    def test_multi_source_exclusion(self):
        """Verify that composite records with flagged components are caught."""
        master_mock = {
            "s1": {"review_status": "approved", "quality_flags": []},
            "s2": {"review_status": "approved", "quality_flags": ["heading_is_paragraph_suspected"]},
            "s3": {"review_status": "ambiguous", "quality_flags": []},
            "s4": {"review_status": "approved", "quality_flags": ["fragment_heading_suspected"]}
        }
        # Both clean
        eligible, reasons = check_eligibility_every_source(["s1"], master_mock)
        self.assertTrue(eligible)
        self.assertEqual(len(reasons), 0)

        # Composite with one flagged component
        eligible, reasons = check_eligibility_every_source(["s1", "s2"], master_mock)
        self.assertFalse(eligible)
        self.assertIn("heading_is_paragraph_suspected", reasons)

        # Ambiguous component
        eligible, reasons = check_eligibility_every_source(["s1", "s3"], master_mock)
        self.assertFalse(eligible)
        self.assertIn("ambiguous", reasons)

        # Fragment heading
        eligible, reasons = check_eligibility_every_source(["s4"], master_mock)
        self.assertFalse(eligible)
        self.assertIn("fragment_heading_suspected", reasons)


class TestPhase3TrainV4Safety(unittest.TestCase):
    def test_train_v4_safety_barriers(self):
        """Verify train_v4 refuses to write to checkpoints/, data/, or non-empty directories."""
        # 1. Writing to checkpoints/
        with self.assertRaises(PermissionError):
            validate_safety(WORKSPACE_ROOT / "checkpoints" / "section_classifier_v4")

        # 2. Writing to data/
        with self.assertRaises(PermissionError):
            validate_safety(WORKSPACE_ROOT / "data" / "bad_dir")

        # 3. Writing to non-empty dir without resume/dry_run
        test_dir = WORKSPACE_ROOT / "runs" / "_test_safety_non_empty"
        test_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = test_dir / "dummy.txt"
        dummy_file.write_text("hello", encoding="utf-8")
        try:
            with self.assertRaises(FileExistsError):
                validate_safety(test_dir, is_dry_run=False, resume=False)
        finally:
            shutil.rmtree(test_dir)

    def test_train_v4_has_no_test_arguments(self):
        """Verify train_v4 argument parser rejects any --test arguments."""
        import train_v4
        parser = train_v4.parse_args
        # Inspect argument strings
        # We can check that neither '--test-file' nor '--test-dataset' exists in sys arguments
        code_text = (WORKSPACE_ROOT / "train_v4.py").read_text(encoding="utf-8")
        self.assertNotIn("--test-file", code_text)
        self.assertNotIn("--test-dataset", code_text)
        self.assertNotIn("--eval-test", code_text)

if __name__ == "__main__":
    unittest.main()
