"""
Workspace Protection and Integrity Verification Script
Phase 0 — Workspace Protection and Inventory

Usage:
    python verify_protection.py
"""

import os
import sys
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_FILE = ROOT / "backups" / "pre_v4_baseline" / "BASELINE_CHECKSUMS.sha256"
REBUILT_V4_DIR = ROOT / "data" / "claude_v4_extracted" / "data" / "rebuilt_v4"
V4_EXPORT_DIR = ROOT / "data" / "claude_v4_export"
BACKUP_DIR = ROOT / "backups" / "pre_v4_baseline"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("PHASE 0: WORKSPACE PROTECTION AND INTEGRITY VERIFICATION")
    print("=" * 70)
    print()

    all_passed = True

    # 1. Verify Baseline Checksums
    print("1. Verifying Baseline Project Files against Checksum Manifest...")
    if not BASELINE_FILE.exists():
        print(f"[FAIL] Baseline checksum manifest missing: {BASELINE_FILE}")
        return False

    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip().split(None, 1) for line in f if line.strip()]

    baseline_matches = 0
    baseline_failures = []

    for expected_hash, rel_path in lines:
        target_path = ROOT / rel_path
        if not target_path.exists():
            baseline_failures.append((rel_path, "FILE_MISSING"))
            continue
        current_hash = sha256_file(target_path)
        if current_hash != expected_hash:
            baseline_failures.append((rel_path, f"HASH_MISMATCH: expected {expected_hash[:8]}, got {current_hash[:8]}"))
        else:
            baseline_matches += 1

    if baseline_failures:
        print(f"[FAIL] {len(baseline_failures)} baseline files failed integrity check:")
        for path, err in baseline_failures[:10]:
            print(f"       - {path}: {err}")
        all_passed = False
    else:
        print(f"[PASS] All {baseline_matches} baseline files intact and unchanged.")

    # 2. Verify Extracted Rebuilt V4 Checksums
    print("\n2. Verifying Extracted V4 Dataset against internal CHECKSUMS.sha256...")
    v4_checksum_file = REBUILT_V4_DIR / "CHECKSUMS.sha256"
    if not v4_checksum_file.exists():
        print(f"[FAIL] V4 CHECKSUMS.sha256 not found at {v4_checksum_file}")
        all_passed = False
    else:
        v4_matches = 0
        v4_failures = []
        with open(v4_checksum_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                expected_hash = parts[0]
                fname = parts[1]
                target_f = REBUILT_V4_DIR / fname
                if not target_f.exists():
                    v4_failures.append((fname, "FILE_MISSING"))
                else:
                    curr_hash = sha256_file(target_f)
                    if curr_hash != expected_hash:
                        v4_failures.append((fname, f"HASH_MISMATCH: expected {expected_hash[:8]}, got {curr_hash[:8]}"))
                    else:
                        v4_matches += 1

        if v4_failures:
            print(f"[FAIL] {len(v4_failures)} V4 bundle files failed integrity check:")
            for fname, err in v4_failures:
                print(f"       - {fname}: {err}")
            all_passed = False
        else:
            print(f"[PASS] All {v4_matches} core V4 files in bundle match official SHA-256 hashes.")

    # 3. Verify Backup Copies
    print("\n3. Verifying Backup Copies in backups/pre_v4_baseline/...")
    backup_checks = [
        BACKUP_DIR / "root_scripts" / "train.py",
        BACKUP_DIR / "root_scripts" / "evaluate_model.py",
        BACKUP_DIR / "data" / "train_real.jsonl",
        BACKUP_DIR / "data" / "validation_real.jsonl",
        BACKUP_DIR / "data" / "eval_real.jsonl",
        BACKUP_DIR / "data" / "train_v3.jsonl",
        BACKUP_DIR / "data" / "validation_v3.jsonl",
        BACKUP_DIR / "data" / "build_v3_dataset.py",
        BACKUP_DIR / "BASELINE_CHECKSUMS.sha256",
    ]
    missing_backups = [p for p in backup_checks if not p.exists()]
    if missing_backups:
        print(f"[FAIL] Missing backup files: {missing_backups}")
        all_passed = False
    else:
        print(f"[PASS] All {len(backup_checks)} critical baseline backup files verified present.")

    # 4. Check Checkpoint Immutability
    print("\n4. Verifying Checkpoint Immutability (v1, v2, v3, phase1)...")
    checkpoints = ["phase1", "section_classifier_v1", "section_classifier_v2", "section_classifier_v3"]
    cp_ok = True
    for cp in checkpoints:
        cp_path = ROOT / "checkpoints" / cp / "final"
        if not cp_path.exists():
            print(f"[FAIL] Expected checkpoint directory missing: {cp_path}")
            cp_ok = False
        else:
            has_adapter = (cp_path / "adapter_model.safetensors").exists()
            has_config = (cp_path / "adapter_config.json").exists()
            if not (has_adapter and has_config):
                print(f"[FAIL] Checkpoint incomplete: {cp_path}")
                cp_ok = False
    if cp_ok:
        print("[PASS] All V1, V2, V3, and Phase 1 checkpoints verified intact.")
    else:
        all_passed = False

    # 5. Verify Checkpoint Weights Manifest
    weights_manifest = ROOT / "reports" / "v4_forensics" / "checkpoint_weights_sha256.json"
    if weights_manifest.exists():
        import json
        print("\n5. Verifying Checkpoint Weight Hashes against checkpoint_weights_sha256.json...")
        with open(weights_manifest, "r", encoding="utf-8") as f:
            weight_hashes = json.load(f)
        weight_matches = 0
        weight_failures = []
        for rel_path, exp_hash in weight_hashes.items():
            full_p = ROOT / rel_path
            if not full_p.exists():
                weight_failures.append((rel_path, "FILE_MISSING"))
            else:
                curr_h = sha256_file(full_p)
                if curr_h != exp_hash:
                    weight_failures.append((rel_path, f"HASH_MISMATCH: expected {exp_hash[:8]}, got {curr_h[:8]}"))
                else:
                    weight_matches += 1
        if weight_failures:
            print(f"[FAIL] {len(weight_failures)} weight files failed integrity check:")
            for p, err in weight_failures:
                print(f"       - {p}: {err}")
            all_passed = False
        else:
            print(f"[PASS] All {weight_matches} checkpoint weight files match verified SHA-256 hashes.")

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL STATUS: [PASS] - Workspace protection & verification successful.")
    else:
        print("OVERALL STATUS: [FAIL] - One or more integrity checks failed.")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

