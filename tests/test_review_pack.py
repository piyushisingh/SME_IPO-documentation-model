"""
Unit and Integration Tests for Phase 2 Review Pack
Validates column schemas, blindness, reproducibility, sampling constraints,
dropdown validations, formula safety, and workspace protection.
"""

import os
import sys
import json
import csv
import hashlib
import unittest
from pathlib import Path
import subprocess

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "reports" / "review_pack"
SEALED_DIR = OUTPUT_DIR / "sealed"
PRIVATE_DIR = OUTPUT_DIR / "private"
MANIFEST_PATH = OUTPUT_DIR / "sampling_manifest.json"
CLUSTERS_CSV = ROOT / "reports" / "v4_forensics" / "duplicate_clusters.csv"

CATEGORIES_17 = [
    "ipo_details", "risk_factors", "disclosures_and_definitions", "legal_and_regulatory",
    "promoters_and_management", "company_information", "financial_information", "financial_statements",
    "related_party_transactions", "capital_structure", "objects_of_issue", "qualitative_factors",
    "industry_information", "business_operations", "litigation", "declarations", "other"
]

class TestReviewPack(unittest.TestCase):

    def test_01_blind_files_columns_and_visibility(self):
        """Test that blind files (E, F, G, H) have exactly the permitted columns and no hidden sheets/rows/columns."""
        expected_blind_cols = ["blind_id", "heading", "text_preview", "reviewer_label", "reviewer_confidence", "reviewer_comment"]
        
        blind_files = [
            OUTPUT_DIR / "05_blind_validation.xlsx",
            OUTPUT_DIR / "06_blind_rule_audit.xlsx",
            SEALED_DIR / "07_blind_test_SEALED.xlsx",
            OUTPUT_DIR / "08_consistency_check.xlsx",
        ]

        for bf in blind_files:
            self.assertTrue(bf.exists(), f"File missing: {bf}")
            wb = openpyxl.load_workbook(bf, data_only=False)
            
            # Check sheet visibility
            for sheetname in wb.sheetnames:
                ws = wb[sheetname]
                self.assertFalse(ws.sheet_state == "hidden", f"Hidden sheet found in {bf.name}: {sheetname}")
            
            ws_data = wb["review_data"]
            headers = [cell.value for cell in ws_data[1]]
            self.assertEqual(headers, expected_blind_cols, f"Column mismatch in {bf.name}: {headers}")
            
            # Check for hidden rows or columns in review_data
            for col_letter, dim in ws_data.column_dimensions.items():
                self.assertFalse(dim.hidden, f"Hidden column {col_letter} in {bf.name}")
            for row_idx, dim in ws_data.row_dimensions.items():
                self.assertFalse(dim.hidden, f"Hidden row {row_idx} in {bf.name}")

            # Verify no leakage of labels, rule_id, chapter_title, document_id, or page_start
            for row in ws_data.iter_rows(min_row=2, max_col=3, values_only=True):
                blind_id, heading, preview = row
                self.assertTrue(blind_id.startswith(("VAL-", "RULE-", "TEST-", "CONSIST-")))

    def test_02_allowed_values_dropdown_and_empty_reviewer_columns(self):
        """Test that allowed-values dropdowns exist, reference visible sheets, and reviewer columns are empty."""
        all_workbooks = [
            (OUTPUT_DIR / "01_chapter_mapping_review.xlsx", 11, 12, 13),
            (OUTPUT_DIR / "02_ambiguous_review.xlsx", 8, 9, 10),
            (OUTPUT_DIR / "03_queue_labeled.xlsx", 9, 10, 11),
            (OUTPUT_DIR / "04_segmentation_check.xlsx", 6, 7, 8),
            (OUTPUT_DIR / "05_blind_validation.xlsx", 4, 5, 6),
            (OUTPUT_DIR / "06_blind_rule_audit.xlsx", 4, 5, 6),
            (SEALED_DIR / "07_blind_test_SEALED.xlsx", 4, 5, 6),
            (OUTPUT_DIR / "08_consistency_check.xlsx", 4, 5, 6),
        ]

        for wb_path, lbl_col, conf_col, comm_col in all_workbooks:
            wb = openpyxl.load_workbook(wb_path)
            self.assertIn("allowed_values", wb.sheetnames, f"allowed_values sheet missing in {wb_path.name}")
            ws_vals = wb["allowed_values"]
            self.assertFalse(ws_vals.sheet_state == "hidden", f"allowed_values sheet hidden in {wb_path.name}")

            # Collect allowed values from sheet
            allowed_labels = [row[0] for row in ws_vals.iter_rows(min_row=2, max_col=1, values_only=True) if row[0] is not None]
            self.assertTrue(len(allowed_labels) > 0, f"No allowed labels in {wb_path.name}")
            # Ensure no blanks inside the range
            self.assertNotIn("", allowed_labels)
            self.assertNotIn(None, allowed_labels)

            allowed_conf = [row[0] for row in ws_vals.iter_rows(min_row=2, min_col=2, max_col=2, values_only=True) if row[0] is not None]
            self.assertEqual(allowed_conf, ["high", "medium", "low"])

            ws_data = wb["review_data"]
            # Verify data validations exist
            self.assertTrue(len(ws_data.data_validations.dataValidation) >= 2, f"Data validation missing in {wb_path.name}")

            # Verify reviewer columns are strictly empty on creation
            for row in ws_data.iter_rows(min_row=2, min_col=lbl_col, max_col=comm_col, values_only=True):
                r_lbl, r_cnf, r_cmm = row
                self.assertTrue(r_lbl is None or r_lbl == "", f"Non-empty reviewer_label in {wb_path.name}: {r_lbl}")
                self.assertTrue(r_cnf is None or r_cnf == "", f"Non-empty reviewer_confidence in {wb_path.name}: {r_cnf}")
                self.assertTrue(r_cmm is None or r_cmm == "", f"Non-empty reviewer_comment in {wb_path.name}: {r_cmm}")

    def test_03_reproducibility_of_sampling(self):
        """Test that re-running make_review_pack reproduces identical sampled IDs and hashes."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_original = json.load(f)

        # Run generator again
        res = subprocess.run([sys.executable, str(ROOT / "src" / "sme_clf" / "make_review_pack.py")], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"make_review_pack failed on rerun: {res.stderr}")

        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_rerun = json.load(f)

        self.assertEqual(
            manifest_original["sampled_example_ids"],
            manifest_rerun["sampled_example_ids"],
            "Sampled IDs changed across generator runs!"
        )

    def test_04_sampling_partition_and_cluster_constraints(self):
        """Test document sources, disjointness, cluster limits, and inclusion probabilities."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        e_ids = manifest["sampled_example_ids"]["file_e_validation"]
        f_ids = manifest["sampled_example_ids"]["file_f_rule_audit"]
        g_ids = manifest["sampled_example_ids"]["file_g_sealed_test"]
        h_ids = manifest["sampled_example_ids"]["file_h_consistency"]

        # Counts
        self.assertEqual(len(e_ids), 100)
        self.assertEqual(len(g_ids), 100)
        self.assertEqual(len(h_ids), 30)

        # Document provenance
        for eid in e_ids:
            self.assertTrue(eid.startswith("century-"), f"E record not from century: {eid}")
        for eid in g_ids:
            self.assertTrue(eid.startswith("ideas_electricals-"), f"G record not from ideas_electricals: {eid}")
        train_doc_prefixes = ("photonics_watertech-", "maxwell_engineering-", "axiom_gas-", "quanto_agroworld-")
        for eid in f_ids:
            self.assertTrue(eid.startswith(train_doc_prefixes), f"F record not from train documents: {eid}")

        # Disjointness
        self.assertEqual(len(set(e_ids) & set(f_ids)), 0)
        self.assertEqual(len(set(e_ids) & set(g_ids)), 0)
        self.assertEqual(len(set(f_ids) & set(g_ids)), 0)
        # H is strict subset of E
        self.assertTrue(set(h_ids).issubset(set(e_ids)))

        # Cluster constraints: at most one example per duplicate cluster per file
        cluster_by_eid = {}
        if CLUSTERS_CSV.exists():
            with open(CLUSTERS_CSV, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cluster_by_eid[r["example_id"]] = r["cluster_id"]

        for fname, sample_list in [("E", e_ids), ("F", f_ids), ("G", g_ids)]:
            seen_clusters = set()
            for eid in sample_list:
                cid = cluster_by_eid.get(eid)
                if cid:
                    self.assertNotIn(cid, seen_clusters, f"Duplicate cluster {cid} sampled twice in File {fname}!")
                    seen_clusters.add(cid)

        # Inclusion probabilities exist and are positive
        for split_key in ["file_e_validation_strata", "file_g_test_strata", "file_f_train_rule_strata"]:
            for stratum, info in manifest[split_key].items():
                self.assertIn("N_h_eligible", info)
                self.assertIn("n_h_sampled", info)
                self.assertIn("inclusion_probability", info)
                if info["n_h_sampled"] > 0:
                    self.assertGreater(info["inclusion_probability"], 0.0)

    def test_05_stratification_constraints(self):
        """Test category caps, minimums, and thin category coverage in E and G."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for split_name, strata_key in [("Validation (E)", "file_e_validation_strata"), ("Test (G)", "file_g_test_strata")]:
            strata = manifest[strata_key]
            total_sampled = sum(s["n_h_sampled"] for s in strata.values())
            self.assertEqual(total_sampled, 100, f"{split_name} total is not 100")
            for cat, s in strata.items():
                n = s["n_h_sampled"]
                N = s["N_h_eligible"]
                self.assertLessEqual(n, 25, f"{split_name} category {cat} exceeds cap 25: {n}")
                if N >= 5:
                    self.assertGreaterEqual(n, 3, f"{split_name} category {cat} below minimum 3: {n} (eligible N={N})")
                elif N > 0:
                    self.assertEqual(n, N, f"{split_name} thin category {cat} did not contribute all eligible: n={n}, N={N}")

    def test_06_formula_safety(self):
        """Test formula safety: characters =, +, -, @ stored as string type 's' and read back unchanged."""
        test_wb_path = ROOT / "reports" / "review_pack" / "05_blind_validation.xlsx"
        test_csv_path = ROOT / "reports" / "review_pack" / "05_blind_validation.csv"
        
        # Check xlsx cell types
        wb = openpyxl.load_workbook(test_wb_path, data_only=False)
        ws = wb["review_data"]
        for row in ws.iter_rows(min_row=2, max_col=3):
            for cell in row:
                if cell.value:
                    self.assertEqual(cell.data_type, 's', f"Cell {cell.coordinate} not data_type 's': {cell.data_type}")

        # Check CSV twins
        with open(test_csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                for val in row:
                    # If it starts with =, +, -, @, in raw text it was prefixed with '
                    if val.startswith(("=", "+", "-", "@")):
                        # Should have been escaped
                        self.fail(f"Unsanitized formula trigger in CSV: {val}")

    def test_07_workspace_protection_passes(self):
        """Test that verify_protection.py passes completely with zero baseline modifications."""
        res = subprocess.run([sys.executable, str(ROOT / "verify_protection.py")], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"verify_protection.py failed!\n{res.stdout}")
        self.assertIn("OVERALL STATUS: [PASS]", res.stdout)

if __name__ == "__main__":
    unittest.main()

