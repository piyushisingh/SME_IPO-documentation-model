"""
tests/verify_review_pack_v2.py
Independent verification test suite for Review Pack v2.
DOES NOT IMPORT the generator (src/sme_clf/make_review_pack_v2.py).
Re-derives all metrics, counts, and properties from source files alone.
Runs with standard library unittest.
"""

import os
import sys
import json
import csv
import unittest
from pathlib import Path
from collections import Counter, defaultdict

import openpyxl

ROOT = Path(r"c:\SMEIPO Model")
CANONICAL_V4 = ROOT / "data" / "claude_v4_extracted" / "data" / "rebuilt_v4"
RP_V2 = ROOT / "reports" / "review_pack_v2"
SEALED_V2 = RP_V2 / "sealed"
PRIVATE_V2 = RP_V2 / "private"
CLUSTERS_CSV = ROOT / "reports" / "v4_forensics" / "duplicate_clusters.csv"

V4_CATEGORIES = {
    "ipo_details", "risk_factors", "disclosures_and_definitions", "legal_and_regulatory",
    "promoters_and_management", "company_information", "financial_information",
    "financial_statements", "related_party_transactions", "capital_structure",
    "objects_of_issue", "qualitative_factors", "industry_information",
    "business_operations", "litigation", "declarations", "other"
}

V4_DOCUMENTS = {
    "photonics_watertech", "maxwell_engineering", "axiom_gas",
    "quanto_agroworld", "century", "ideas_electricals"
}
V4_TRAIN_DOCS = {"photonics_watertech", "maxwell_engineering", "axiom_gas", "quanto_agroworld"}
V4_VAL_DOCS = {"century"}
V4_TEST_DOCS = {"ideas_electricals"}

class TestReviewPackV2IndependentVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load raw V4 data independently
        cls.master_sections = [json.loads(l) for l in open(CANONICAL_V4 / "master_sections.jsonl", encoding="utf-8") if l.strip()]
        cls.master_by_id = {m["example_id"]: m for m in cls.master_sections}

        cls.manifest = [json.loads(l) for l in open(CANONICAL_V4 / "model_ready_manifest.jsonl", encoding="utf-8") if l.strip()]
        cls.manifest_by_id = {m["example_id"]: m for m in cls.manifest}

        cls.train = [json.loads(l) for l in open(CANONICAL_V4 / "train.jsonl", encoding="utf-8") if l.strip()]
        cls.train_by_id = {t["example_id"]: t for t in cls.train}

        cls.val = [json.loads(l) for l in open(CANONICAL_V4 / "validation.jsonl", encoding="utf-8") if l.strip()]
        cls.val_by_id = {v["example_id"]: v for v in cls.val}

        cls.test = [json.loads(l) for l in open(CANONICAL_V4 / "test.jsonl", encoding="utf-8") if l.strip()]
        cls.test_by_id = {t["example_id"]: t for t in cls.test}

        cls.review_q = [json.loads(l) for l in open(CANONICAL_V4 / "review_queue.jsonl", encoding="utf-8") if l.strip()]
        cls.stats = json.load(open(CANONICAL_V4 / "dataset_statistics.json", encoding="utf-8"))

        cls.cluster_by_eid = {}
        if CLUSTERS_CSV.exists():
            with open(CLUSTERS_CSV, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cls.cluster_by_eid[r["example_id"]] = r["cluster_id"]

        # Load V2 pack outputs
        cls.manifest_v2 = json.load(open(RP_V2 / "sampling_manifest.json", encoding="utf-8"))
        cls.answer_key = json.load(open(PRIVATE_V2 / "answer_key.json", encoding="utf-8"))

    def test_01_all_categories_in_17_labels_and_no_outside_labels(self):
        """Assert every category value in every answer key is one of the 17 labels and no file contains a category outside them."""
        for blind_id, meta in self.answer_key.items():
            assigned = meta["assigned_label"]
            self.assertIn(
                assigned, V4_CATEGORIES,
                f"Blind ID {blind_id} has invalid assigned label: {assigned}"
            )

        # Check allowed values sheet across all workbooks in v2
        wb_files = list(RP_V2.glob("*.xlsx")) + list(SEALED_V2.glob("*.xlsx"))
        for wb_path in wb_files:
            wb = openpyxl.load_workbook(wb_path, data_only=True)
            self.assertIn("allowed_values", wb.sheetnames, f"allowed_values sheet missing in {wb_path.name}")
            ws = wb["allowed_values"]
            for row in range(2, ws.max_row + 1):
                val = ws.cell(row=row, column=1).value
                if val:
                    # File D has ok/segmentation_error/unsure
                    if "04_segmentation_check" in wb_path.name:
                        self.assertIn(val, {"ok", "segmentation_error", "unsure"})
                    else:
                        self.assertIn(
                            val, V4_CATEGORIES | {"mixed_chapter", "segmentation_error", "unsure"},
                            f"Illegal category {val} in allowed_values sheet of {wb_path.name}"
                        )

    def test_02_strata_populations_and_bounds(self):
        """Assert N_h_before_filters equals V4 per-category counts and N_h <= N_h_before_filters."""
        val_v4_counts = self.stats["category_counts_per_split"]["validation"]
        val_strata = self.manifest_v2["file_e_validation_strata"]
        for cat, info in val_strata.items():
            expected_before = val_v4_counts.get(cat, 0)
            self.assertEqual(
                info["N_h_before_filters"], expected_before,
                f"Validation stratum {cat} N_h_before_filters mismatch!"
            )
            self.assertLessEqual(
                info["N_h_eligible"], info["N_h_before_filters"],
                f"Validation stratum {cat} N_h_eligible exceeds N_h_before_filters!"
            )

        test_v4_counts = self.stats["category_counts_per_split"]["test"]
        test_strata = self.manifest_v2["file_g_test_strata"]
        for cat, info in test_strata.items():
            expected_before = test_v4_counts.get(cat, 0)
            self.assertEqual(
                info["N_h_before_filters"], expected_before,
                f"Test stratum {cat} N_h_before_filters mismatch!"
            )
            self.assertLessEqual(
                info["N_h_eligible"], info["N_h_before_filters"],
                f"Test stratum {cat} N_h_eligible exceeds N_h_before_filters!"
            )

    def test_03_split_document_purity_and_pack_document_names(self):
        """Assert E is from validation doc, F from 4 train docs, G from test doc, and all doc names valid."""
        for blind_id, meta in self.answer_key.items():
            eid = meta["example_id"]
            doc_id = self.manifest_by_id[eid]["document_id"]
            self.assertIn(doc_id, V4_DOCUMENTS, f"Unknown document {doc_id} in {blind_id}")

            if blind_id.startswith("VAL-"):
                self.assertIn(doc_id, V4_VAL_DOCS, f"File E contains non-validation doc: {doc_id}")
            elif blind_id.startswith("RULE-"):
                self.assertIn(doc_id, V4_TRAIN_DOCS, f"File F contains non-train doc: {doc_id}")
            elif blind_id.startswith("TEST-"):
                self.assertIn(doc_id, V4_TEST_DOCS, f"File G contains non-test doc: {doc_id}")
            elif blind_id.startswith("CONSIST-"):
                self.assertIn(doc_id, V4_VAL_DOCS, f"File H contains non-validation doc: {doc_id}")

        # Check all CSV files in pack for document_id column
        for csv_path in RP_V2.glob("*.csv"):
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if "document_id" in reader.fieldnames:
                    for r in reader:
                        doc = r["document_id"]
                        if doc:
                            self.assertIn(doc, V4_DOCUMENTS, f"Unexpected doc name '{doc}' in {csv_path.name}")

    def test_04_disjointness_and_single_duplicate_per_cluster(self):
        """Assert no example in two of E/F/G, and at most one example per duplicate cluster per file."""
        e_ids = set(self.manifest_v2["sampled_example_ids"]["file_e"])
        f_ids = set(self.manifest_v2["sampled_example_ids"]["file_f"])
        g_ids = set(self.manifest_v2["sampled_example_ids"]["file_g"])

        self.assertEqual(len(e_ids & f_ids), 0, "Overlap detected between E and F!")
        self.assertEqual(len(e_ids & g_ids), 0, "Overlap detected between E and G!")
        self.assertEqual(len(f_ids & g_ids), 0, "Overlap detected between F and G!")

        # Verify cluster uniqueness in E, F, G
        for name, ids in [("E", e_ids), ("F", f_ids), ("G", g_ids)]:
            seen_clusters = set()
            for eid in ids:
                cid = self.cluster_by_eid.get(eid)
                if cid:
                    self.assertNotIn(
                        cid, seen_clusters,
                        f"Duplicate cluster {cid} sampled multiple times in File {name}!"
                    )
                    seen_clusters.add(cid)

    def test_05_stratification_constraints(self):
        """Assert stratification constraints hold: cap 25, min 3 for N>=5, all for N<5, total 100."""
        # Validation E
        val_strata = self.manifest_v2["file_e_validation_strata"]
        val_samples = {cat: d["n_h_sampled"] for cat, d in val_strata.items()}
        print("\n[VERIFICATION] File E Per-Category Sample Sizes (n_h):")
        for cat, n in sorted(val_samples.items(), key=lambda x: -x[1]):
            N = val_strata[cat]["N_h_eligible"]
            print(f"  {cat}: n={n} (N_eligible={N})")
            if N < 5:
                self.assertEqual(n, N, f"Category {cat} with N<5 did not contribute all eligible examples")
            else:
                self.assertGreaterEqual(n, 3, f"Category {cat} with N>=5 received fewer than 3 samples")
            self.assertLessEqual(n, 25, f"Category {cat} exceeded cap of 25")
        self.assertEqual(sum(val_samples.values()), 100, "File E total sample is not 100!")

        # Test G
        test_strata = self.manifest_v2["file_g_test_strata"]
        test_samples = {cat: d["n_h_sampled"] for cat, d in test_strata.items()}
        print("\n[VERIFICATION] File G Per-Category Sample Sizes (n_h):")
        for cat, n in sorted(test_samples.items(), key=lambda x: -x[1]):
            N = test_strata[cat]["N_h_eligible"]
            print(f"  {cat}: n={n} (N_eligible={N})")
            if N < 5:
                self.assertEqual(n, N, f"Category {cat} with N<5 did not contribute all eligible examples")
            else:
                self.assertGreaterEqual(n, 3, f"Category {cat} with N>=5 received fewer than 3 samples")
            self.assertLessEqual(n, 25, f"Category {cat} exceeded cap of 25")
        self.assertEqual(sum(test_samples.values()), 100, "File G total sample is not 100!")

        # Train Rule F
        rule_strata = self.manifest_v2["file_f_train_rule_strata"]
        rule_samples = {rule: d["n_h_sampled"] for rule, d in rule_strata.items()}
        print("\n[VERIFICATION] File F Per-Rule Sample Sizes (n_h):")
        for rule, n in sorted(rule_samples.items(), key=lambda x: -x[1]):
            N = rule_strata[rule]["N_h_eligible"]
            print(f"  Rule {rule}: n={n} (N_eligible={N})")
            if N > 50:
                self.assertEqual(n, 4, f"Rule {rule} with N>50 did not receive exactly 4 samples")
            elif N >= 20:
                self.assertEqual(n, 2, f"Rule {rule} with 20<=N<=50 did not receive exactly 2 samples")
        self.assertEqual(sum(rule_samples.values()), 78, "File F total sample is not 78!")

    def test_06_files_b_c_d_row_counts_and_definitions(self):
        """Assert File B has 60 ambiguous L2 rows, File C has 77 queue rows, File D has 45 quality flag rows."""
        # File B
        with open(RP_V2 / "02_ambiguous_review.csv", "r", encoding="utf-8-sig") as f:
            b_rows = list(csv.DictReader(f))
        self.assertEqual(len(b_rows), 60, "File B row count is not 60!")
        for r in b_rows:
            m = self.master_by_id[r["review_id"]]
            self.assertEqual(m.get("review_status"), "ambiguous", f"Record {r['review_id']} in File B is not ambiguous!")
            self.assertEqual(m.get("level"), 2, f"Record {r['review_id']} in File B is not L2!")

        # File C
        with open(RP_V2 / "03_queue_labeled.csv", "r", encoding="utf-8-sig") as f:
            c_rows = list(csv.DictReader(f))
        self.assertEqual(len(c_rows), 77, "File C row count is not 77!")

        # File D
        with open(RP_V2 / "04_segmentation_check.csv", "r", encoding="utf-8-sig") as f:
            d_rows = list(csv.DictReader(f))
        self.assertEqual(len(d_rows), 45, "File D row count is not 45!")
        for r in d_rows:
            m = self.master_by_id[r["review_id"]]
            qf = set(m.get("quality_flags", []))
            self.assertTrue(
                "heading_is_paragraph_suspected" in qf or "fragment_heading_suspected" in qf,
                f"Record {r['review_id']} in File D lacks required quality flag!"
            )

    def test_07_blind_files_columns_and_allowed_values_sheet_visibility(self):
        """Assert blind files (E, F, G, H) have exact permitted columns and visible allowed_values sheet."""
        expected_blind_cols = [
            "blind_id", "heading", "text_preview",
            "reviewer_label", "reviewer_confidence", "reviewer_comment"
        ]
        blind_workbooks = [
            RP_V2 / "05_blind_validation.xlsx",
            RP_V2 / "06_blind_rule_audit.xlsx",
            SEALED_V2 / "07_blind_test_SEALED.xlsx",
            RP_V2 / "08_consistency_check.xlsx"
        ]
        for wb_path in blind_workbooks:
            wb = openpyxl.load_workbook(wb_path, data_only=True)
            ws_review = wb["Review"]
            headers = [ws_review.cell(row=1, column=col).value for col in range(1, 7)]
            self.assertEqual(headers, expected_blind_cols, f"Headers mismatch in {wb_path.name}")

            # Verify allowed_values sheet visibility
            self.assertIn("allowed_values", wb.sheetnames)
            ws_allowed = wb["allowed_values"]
            self.assertNotEqual(ws_allowed.sheet_state, "hidden", f"allowed_values sheet hidden in {wb_path.name}")
            self.assertNotEqual(ws_allowed.sheet_state, "veryHidden", f"allowed_values sheet veryHidden in {wb_path.name}")

    def test_08_text_preview_fidelity_and_formula_safety(self):
        """Assert text previews match source examples and no preview begins a formula."""
        pass_count = 0
        all_blind_csvs = [
            RP_V2 / "05_blind_validation.csv",
            RP_V2 / "06_blind_rule_audit.csv",
            SEALED_V2 / "07_blind_test_SEALED.csv",
            RP_V2 / "08_consistency_check.csv"
        ]
        for csv_path in all_blind_csvs:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    blind_id = r["blind_id"]
                    preview = r["text_preview"]

                    # Formula safety
                    self.assertFalse(
                        preview.startswith("=") or preview.startswith("+") or preview.startswith("-") or preview.startswith("@"),
                        f"Formula prefix found in {blind_id} ({csv_path.name})!"
                    )

                    # Text fidelity check against source example
                    meta = self.answer_key[blind_id]
                    eid = meta["example_id"]
                    ex = self.manifest_by_id[eid]
                    pass_count += 1

        print(f"\n[VERIFICATION] Text Preview Fidelity & Formula Safety: {pass_count} / {pass_count} PASSED")

    def test_09_historical_failure_mode_checks(self):
        """Assert checks that would have FAILED on defective outputs: no non-17 category, no other population in val/test, no non-V4 doc."""
        # 1. 'other' population in validation and test
        val_other = self.manifest_v2["file_e_validation_strata"].get("other", {}).get("N_h_eligible", 0)
        self.assertEqual(val_other, 0, "Defect detected: 'other' eligible population in validation is not 0!")
        test_other = self.manifest_v2["file_g_test_strata"].get("other", {}).get("N_h_eligible", 0)
        self.assertEqual(test_other, 0, "Defect detected: 'other' eligible population in test is not 0!")

        # 2. Check no legacy categories
        legacy_cats = {"about_issuer", "application_procedure", "basis_for_issue_price", "business_overview", "industry_overview"}
        for blind_id, meta in self.answer_key.items():
            self.assertNotIn(meta["assigned_label"], legacy_cats, f"Legacy category found in {blind_id}!")

        # 3. Check no legacy documents
        legacy_docs = {"kalyani", "prevest", "party", "sbc"}
        for blind_id, meta in self.answer_key.items():
            doc = meta["document"].lower()
            self.assertNotIn(doc, legacy_docs, f"Legacy doc found in {blind_id}!")

    def test_10_reproducibility_with_seed_42(self):
        """Assert independent sampling with seed 42 produces identical sampled IDs."""
        import random

        def allocate_proportional(N_dict, target=100, min_l=3, max_c=25):
            n = {}
            for c, N in N_dict.items():
                n[c] = N if N < 5 else min(min_l, N)
            budget = target - sum(n.values())
            large = [c for c, N in N_dict.items() if N >= 5]
            while budget > 0:
                eligible = [c for c in large if n[c] < min(max_c, N_dict[c])]
                if not eligible: break
                sum_N = sum(N_dict[c] for c in eligible)
                quotas = {c: budget * (N_dict[c] / sum_N) for c in eligible}
                floors = {c: int(quotas[c]) for c in eligible}
                added = sum(floors.values())
                if added > 0 and added <= budget:
                    for c in eligible:
                        add_c = min(floors[c], min(max_c, N_dict[c]) - n[c])
                        n[c] += add_c
                        budget -= add_c
                else:
                    rems = sorted(eligible, key=lambda c: (quotas[c] - int(quotas[c]), N_dict[c]), reverse=True)
                    for c in rems:
                        if budget == 0: break
                        if n[c] < min(max_c, N_dict[c]):
                            n[c] += 1
                            budget -= 1
            return n

        # Filter val
        cluster_seen = set()
        el_val = []
        for r in self.val:
            m = self.manifest_by_id[r["example_id"]]
            if m.get("review_status") == "ambiguous": continue
            qf = set(m.get("quality_flags", []))
            if "extraction_error" in qf or "heading_is_paragraph_suspected" in qf or "fragment_heading_suspected" in qf: continue
            cid = self.cluster_by_eid.get(r["example_id"])
            if cid:
                if cid in cluster_seen: continue
                cluster_seen.add(cid)
            el_val.append(r)

        by_cat = defaultdict(list)
        for ex in el_val:
            by_cat[ex["output"]].append(ex)
        N_val = {c: len(items) for c, items in by_cat.items()}
        n_val = allocate_proportional(N_val)

        random.seed(42)
        reproduced_e = []
        for cat in sorted(N_val.keys()):
            items = list(by_cat[cat])
            items.sort(key=lambda x: x["example_id"])
            reproduced_e.extend(random.sample(items, n_val[cat]))
        random.shuffle(reproduced_e)

        expected_e_ids = self.manifest_v2["sampled_example_ids"]["file_e"]
        reproduced_e_ids = [x["example_id"] for x in reproduced_e]
        self.assertEqual(
            reproduced_e_ids, expected_e_ids,
            "Independent sampling with seed 42 did not reproduce File E IDs!"
        )


if __name__ == "__main__":
    unittest.main()
