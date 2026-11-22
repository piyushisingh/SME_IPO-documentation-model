<div align="center">
SME IPO Documentation AI

A document-intelligence pipeline for Indian SME IPO offer documents: structure extraction, section classification, and a rigorously audited fine-tuning and evaluation workflow.

Show Image Show Image Show Image Show Image Show Image Show Image

Status: interim. Dataset and evaluation infrastructure are built and verified. Model evaluation against human-verified labels is still in progress, so no model-quality claim is made here.

</div>
Overview

Indian SME IPO offer documents (DRHP, RHP, and final prospectuses) typically run to 300–400 pages. They follow a common SEBI-prescribed chapter layout, but their content, headings and formatting vary widely between issuers. Finding, comparing, or checking specific disclosures by hand is slow and error-prone.

This project builds the machine-readable foundation for assisting with that work:

Turn an offer-document PDF into structured, page-referenced sections.
Classify each section into one of 17 regulatory and business categories (risk_factors, objects_of_issue, litigation, financial_statements, …).
Provide a trustworthy evaluation workflow so any classifier can be measured honestly.

The section classifier is one component of a larger planned system (retrieval with page citations, grounded question answering, and documentation assistance). This repository does not yet implement retrieval, question answering, document generation, or any application layer, and it does not give legal, financial or investment advice.

A large part of the engineering effort went into the data and evaluation, not only the model. Earlier classifier versions scored poorly, and the audit trail in this repository documents why (truncated training targets, inconsistent prompts, an evaluator that never read most of each section, near-duplicate leakage, and labels that were never human-verified), and what was built to fix each.

Architecture
SME IPO PDFDRHP / RHP / Prospectus
Page-level text extractionOCR only for scanned pages
TOC-anchored chapterdetection+ heading-basedsub-sections
Section datasetpage provenance, raw +cleaned text
Provisional category labelsrule-derived, unverified
Dataset auditduplicates, leakage,truncation, tokenizer checks
Token-aware training datadocument-level splits
Gemma-3-1B-it + LoRATRL SFT, completion-onlyloss
Evaluation harnessconstrained label scoring,ablations, metrics
Classifier outputplanned: confidence +abstention
Retrieval with citationsPLANNED
Grounded Q&A anddraftingPLANNED
Android clientPLANNED
Component	Role	Status
PDF text extraction and section structuring	Page-level extraction, table-of-contents anchoring, sub-section detection, OCR for scanned pages	Implemented (rebuilt dataset pipeline)
Dataset audit and validation	Count/schema checks, duplicate detection, token-length analysis, leakage checks	Implemented and verified
Token-aware dataset builder	Builds training examples so the target label can never be truncated	Implemented and verified
Fine-tuning (train_v4.py)	Config-driven LoRA training with safety guards	Implemented; pipeline dry-runs only, no V4 training run yet
Evaluation harness (evaluate_v4.py)	Exact constrained scoring, views, bootstrap CIs, test-set guard	Implemented; smoke-tested only
Baselines and fair V3 re-evaluation	TF-IDF and rule baselines; V3 measured without contamination	Planned / in progress
Human label verification	Reviewer workbooks and blind samples	Materials built; review in progress
Retrieval, Q&A, generation, Android app	Product layers	Planned
What is in this repository
Document processing and dataset engineering
Extraction: page-level text with provenance (PDF page, printed page), with OCR used only where a page has no usable text layer and recorded separately.
Segmentation: chapters are anchored on each document's own table of contents; sub-sections come from heading detection. Sub-sections partition their chapter exactly, and an independent check confirms that no text is dropped or duplicated.
Dataset versions: V1 to V3 are the original experiments. V4 is a rebuilt candidate dataset from six source documents (DRHP, RHP, fixed-price final prospectus, and an abridged-prospectus part), with 5,997 section records.
Splits are document-level: no document appears in more than one of train / validation / test.
Labels in V4 are rule-derived from table-of-contents chapter titles and have not been human-verified. About 90% of labels are inherited from a chapter rule. A classifier scored against these labels measures agreement with the rules, not correctness. A human review pack (chapter-mapping audit, blind validation sample, blind rule audit) was built to address this; see Limitations.
Source documents and extracted prospectus text are not distributed in this repository. You must supply your own offer documents.
Model training
Item	Setting	Source
Base model	unsloth/gemma-3-1b-it (4-bit via bitsandbytes), used from the local Hugging Face cache	train.py, train_v4.py
Adaptation	LoRA, r = 16, alpha = 32	train.py, configs/
Trainer	TRL SFTTrainer, completion_only_loss=True	train.py
Effective batch size	8 (V3: 2 × 4; V4 dry-runs: 1 × 8)	reports/phase3/
Hardware	NVIDIA RTX 4060 Laptop GPU (8 GB)	env/

Using or redistributing Gemma-based weights requires accepting Google's Gemma Terms of Use.

V1, V2 and V3 adapters were trained during earlier experiments. No V4 adapter has been trained yet.

Evaluation framework

evaluate_v4.py and src/sme_clf/ provide:

Exact constrained scoring: each of the 17 labels is scored by the sequence log-probability of the label tokens given the prompt, so the model can never emit an invalid label. First-token scoring alone is not sufficient: financial_information and financial_statements share their first token.
Metrics: accuracy, macro-F1 (over classes with support), weighted-F1, per-class precision/recall/F1, confusion matrices, bootstrap confidence intervals with an explicit warning that examples within a document are correlated.
Heading ablation: heading-free evaluation that removes the Heading: line and the heading copy at the start of the body (about 94.5% of V4 bodies repeat the heading, so removing only the prompt line would leak it).
Views: all examples, de-duplicated, boilerplate-only, long vs short.
Frozen test set guard: the test split cannot be evaluated without explicit confirmation flags; every access is logged, and repeat evaluation of the same model is refused.
Verified findings

All values below come from saved reports in reports/ (see the cited folder). They describe the data and pipeline, not model quality.

Finding	Value	Source	Why it matters
V4 dataset size	5,997 master records; 5,362 model-ready examples (3,801 train / 731 validation / 830 test)	reports/v4_forensics/	Reproduced independently from the exported files
Documents	6 source documents; document-level split	reports/v4_forensics/	Too few for generalization claims
Labels inherited from chapter rules	90.31% of master records	reports/v4_forensics/	Labels are provisional, not ground truth
Heading repeated at start of body	about 94.3–94.6% across splits	reports/v4_forensics/	Heading ablation must strip both copies
Near-duplicates with train	56 of 731 validation, 46 of 830 test examples	reports/v4_forensics/	Shared SEBI boilerplate inflates scores on those examples
Plain right-truncation at 768 tokens	would remove the target label from 129 / 3,801 train examples	reports/v4_forensics/	Explains why earlier training could degrade
Old evaluator limit	256 tokens truncates 543 of 830 test prompts; the prompt instruction alone uses 109 tokens	reports/v4_forensics/	Earlier accuracy figures were measured on partial inputs
Token-aware builder	target preserved in 100% of examples at 768 / 1024 / 1536 tokens (0 violations in the audit)	reports/phase3/	Training can no longer lose its target
Body-token retention at 1024	79.3% (train) / 79.6% (validation), with 6.0% / 7.9% of bodies truncated	reports/phase3/truncation_token_retention.json	Quantifies the cost of the sequence cap
Worst-case memory	about 2.9 GB peak GPU memory at 1536 tokens on the 16 longest training examples	reports/phase3/dryrun_v2_results.json	Fits comfortably in 8 GB; timing figures from this run were inconsistent and are not relied on
Double-BOS investigation	V3 training did not use a double BOS token; the earlier warning came from a flaw in an audit script	reports/phase3/v3_bos_verdict.json	Rules out one suspected cause of V3 behavior
Constrained vs naive scoring	identical argmax on 20 / 20 validation examples, max log-probability difference 0.0	reports/phase4/	Harness check only (see below)

What these findings do not show: model accuracy, correctness of the labels, generalization to unseen SME IPOs, or production readiness. The 20-example evaluation above is a harness smoke test; its accuracy figures are deliberately not reported.

<details> <summary><b>Historical experiment log (V1 to V3)</b></summary>
Version	Approach	Recorded outcome	Caveat
V1	Full-sequence loss	about 29% evaluation accuracy (project notes)	Model sometimes generated document fragments instead of labels
V2	Completion-only loss, 3 epochs, 768 tokens	about 33% evaluation accuracy (project notes)	Later audit found many target labels were lost to right-truncation
V3	Token-aware truncation preserving heading and target	Training completed; final accuracy not reliably captured	Evaluated with a 256-token limit; re-evaluation planned. The V3 training data included one of the V4 documents, so V3 cannot be fairly scored on V4's test document

These figures come from project notes, were produced under the data and evaluation problems described above, and are not comparable to each other or to future results.

</details>
Development timeline
Phase	Objective	Evidence	Status
Original pipeline (V1–V3)	Extract sections, label, fine-tune Gemma-3-1B-it	train.py, evaluate_model.py, data/ scripts	Implemented; results unreliable
Dataset rebuild (V4)	Traceable, page-referenced sections with audited splits	reports/v4_forensics/	Implemented; labels unverified
Phase 0–1	Workspace protection (checksums for baseline files and checkpoint weights); forensic validation of V4 claims	verify_protection.py, reports/v4_forensics/	Experimentally validated
Review pack	Reviewer workbooks, blind samples, integrity audit	src/sme_clf/make_review_pack_v2.py, tests/verify_review_pack_v2.py	Implemented; human review in progress
Phase 3	Shared prompt module, token-aware builder, dataset auditor, guarded trainer, dry runs	src/sme_clf/, train_v4.py, reports/phase3/	Implemented; no training run
Phase 4	Evaluation harness with test guard, exact constrained scoring	evaluate_v4.py, reports/phase4/	Implemented; smoke-tested only
Phase 5	Fast exact scoring, baselines, fair V3 re-evaluation, mask check via the real trainer path	—	In progress
Later	Human-verified labels, controlled V4 training, hybrid classifier, retrieval	—	Planned
Repository structure
text
.
├── train.py                  # original LoRA SFT training script (V1–V3)
├── train_v4.py               # config-driven, safety-guarded trainer (V4)
├── evaluate_model.py         # original evaluator (legacy; fixed 256-token limit)
├── evaluate_v4.py            # evaluation harness with test guard
├── verify_protection.py      # integrity check for baseline files and checkpoint weights
├── PROJECT_RULES.md          # binding rules for experiments (immutability, frozen test, reporting)
├── docs/
│   └── PROJECT_STATE.md      # phase-by-phase project log
├── src/sme_clf/              # prompting, dataset builder, auditor, predictors, metrics, review-pack tools
├── configs/                  # training configuration (train_v4_default.json)
├── data/
│   ├── scripts/              # original extraction and dataset-building scripts
│   ├── build_v3_dataset.py   # V3 token-preserving dataset builder
│   └── audit_v3_chat_template.py
├── tests/                    # unittest suites (phase 3, phase 4, review pack)
├── reports/                  # aggregate audit and benchmark reports (no source text)
└── env/                      # environment snapshots (pip freeze, GPU info)

Model checkpoints, backups, credentials, source documents, extracted prospectus text and review materials are intentionally not part of the published tree.

Getting started

Prerequisites: Windows 11 or Linux, Python 3.12, an NVIDIA GPU with CUDA support (developed on an 8 GB RTX 4060 Laptop GPU), and a Hugging Face account for model access.

powershell
git clone https://github.com/piyushisingh/SME_IPO-documentation-model.git
cd SME_IPO-documentation-model
python -m venv unsloth_env
.\unsloth_env\Scripts\Activate.ps1

Dependencies. There is no pinned requirements file. The verified environment is in env/pip_freeze.txt: PyTorch 2.6.0 (CUDA 12.8), Transformers 4.49.0, TRL 0.24.0, PEFT 0.14.0, Datasets 3.3.2, plus Unsloth and bitsandbytes. Install a CUDA-matched PyTorch first, then the remaining packages at those versions.

Hugging Face access. Set your token as an environment variable (for example HF_TOKEN) or in a local .env file that is excluded from version control. Never commit it.

Data. Source PDFs and derived datasets are not included. Build your own section datasets from offer documents you are entitled to use.

powershell
# Integrity check of the protected baseline files and checkpoint weights
python verify_protection.py

# Unit tests
python -m unittest discover -s tests

# Inspect the training and evaluation options (no training or evaluation is run)
python train_v4.py --help
python evaluate_v4.py --help

train_v4.py is driven by configs/train_v4_default.json; it refuses to write into existing checkpoint or data folders and refuses a non-empty output directory. evaluate_v4.py requires an explicit --max-seq-len and refuses the frozen test split unless confirmation flags are supplied. Check --help for the exact options before running either.

Engineering decisions
Token-aware truncation. Only the section body is cut; the heading and the target label are always preserved. Right-truncating a rendered prompt silently deleted the target in earlier versions, and the target is now a separate, never-truncated field with a hard audit.
One prompt module. Training, evaluation and (later) inference share one prompt builder with a single BOS-token policy and a prompt hash, removing train/eval format mismatch.
Constrained label scoring instead of free generation. Free generation produced invalid and fragmentary outputs in earlier versions. Scoring the 17 valid labels exactly removes invalid outputs and yields a probability-like score for later abstention.
Document-level splits and a frozen test set. Section-level random splits leak document style. The test document is never used for tuning, and access is logged.
Duplicate-aware evaluation. SEBI boilerplate recurs across filings. Results can be reported with duplicates included or excluded.
Protection and reproducibility. Checksums cover baseline files and checkpoint weights; runs write their configuration, dataset hashes and library versions to a manifest; seeds are fixed.
Reporting discipline. Every number in a report must trace to a saved output file.
Limitations
No human-verified labels. V4 labels are rule-derived from chapter titles; their error rate is unknown. Scores against them measure agreement with the rules.
Six source documents. Validation is one company's filing and the test set is another's. Results cannot establish performance on unseen SME IPOs.
Small and absent categories. declarations and qualitative_factors have very few examples, and other has none.
Near-duplicate boilerplate appears across splits and can inflate results on those examples.
No trustworthy model-level metric has been established yet. V3 has not been re-evaluated under the corrected harness, and no V4 model has been trained.
Tables are kept as column-aligned text and are not parsed into structured rows.
A fixed-price final prospectus appears only in training data.
Roadmap (all planned)
Complete human review of chapter mappings and blind samples; curate a corrected dataset version.
Compare simple baselines (TF-IDF, rules) against the LoRA classifier on human-verified data.
Controlled V4 training runs with validation-only model selection and a single final test evaluation.
Hybrid classifier: deterministic title rules where reliable, model elsewhere, with confidence and abstention.
Retrieval with page-level citations, grounded question answering, and template-based drafting that never generates unsourced facts.
API and Android client.
Responsible use

Outputs are intended for organizing and navigating documents. They can be wrong and must be verified by a qualified person against the source filing. This project is not legal, financial or investment advice. Source documents, private evaluation material, credentials and answer keys are not published. Respect the usage rights of any offer documents you process.
