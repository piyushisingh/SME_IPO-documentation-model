# SME IPO Section Classification — Reviewer Guide

This guide defines the 17 document categories and core decision principles for human audit and review of Indian SME IPO prospectus sections (DRHP / RHP / Prospectus).

---

## 1. Core Operating Principle
> **"Label by what the text IS, not by what you expect its chapter to be."**

SME IPO prospectuses frequently combine disparate topics inside standardized chapters (e.g., litigation summaries inside *General Information*, or underwriter bios inside *Other Regulatory Disclosures*). Always classify based on the **actual substance and purpose of the section text**, even if it conflicts with the parent chapter title.

---

## 2. Category Definitions (Condensed)

1. **`ipo_details`**: Mechanics, structure, and terms of the offering. Issue price, price band, lot size, bid/issue period, issue procedure, basis of allotment, underwriting, syndicate members, registrar, book running lead manager (BRLM), listing exchanges, credit ratings, monitoring agency.
2. **`risk_factors`**: Explicit disclosures of risks affecting the company, promoters, industry, or securities. Internal risks, external/macro risks, operational risks, risk factor numbering, materiality thresholds.
3. **`disclosures_and_definitions`**: Conventions, glossary, abbreviations, industry terms, definitions, and front-matter explanatory notes.
4. **`legal_and_regulatory`**: Statutory authorizations, approvals from regulatory authorities (SEBI, RBI, ROC, Stock Exchanges), compliance status, corporate governance regulations, labor/environmental law compliances.
5. **`promoters_and_management`**: Biographies, work profiles, shareholding, interests, compensation, and appointments of promoters, board of directors, key managerial personnel (KMP), and senior management.
6. **`company_information`**: History, corporate structure, date of incorporation, changes in registered office, main objects, memorandum and articles of association (MOA/AOA), subsidiaries, joint ventures, acquisitions.
7. **`financial_information`**: Narrative financial discussion, Management Discussion and Analysis (MD&A), capital expenditure plans, financial metrics, liquidity, cash flow analysis, results of operations.
8. **`financial_statements`**: Formal restated financial statements, balance sheets, profit & loss statements, cash flow statements, notes to accounts, auditor restatement examination reports, accounting policies.
9. **`related_party_transactions`**: Contracts, arrangements, borrowings, guarantees, or dealings between the company and promoters, directors, subsidiaries, or entities under common control.
10. **`capital_structure`**: Equity share capital, authorized/issued/subscribed/paid-up capital, history of share capital, bonus issues, preferential allotments, employee stock option schemes (ESOPs), promoter lock-in schedules.
11. **`objects_of_issue`**: Fund utilization, purpose of the fresh issue/offer for sale, schedule of implementation, deployment of funds, bridge financing, means of finance.
12. **`qualitative_factors`**: Qualitative statements describing competitive strengths, business strategies, key performance differentiators, and strategic focus areas.
13. **`industry_information`**: Macroeconomic overview, sector analysis, competitive landscape, market research, industry drivers, growth projections (often citing CRISIL, CareEdge, Mordor, etc.).
14. **`business_operations`**: Products, services, manufacturing facilities, operational capacity, raw materials, customers, suppliers, sales/marketing, research & development (R&D), quality control.
15. **`litigation`**: Pending civil, criminal, tax, labor, or regulatory legal proceedings against the company, promoters, directors, or subsidiaries; material developments and contingent liabilities.
16. **`declarations`**: Statutory certificates, promoter/director declarations of truthfulness, auditor consent letters, legal counsel consents, sign-offs under Companies Act / SEBI ICDR.
17. **`other`**: Procedural front/back matter that does not fit any of the 16 functional categories (e.g., table of contents, bibliography, document dispatch details).

---

## 3. Boundary & Overlap Rules
- **MD&A vs. Financial Statements:** Narrative discussion of performance is `financial_information`. The tabulated balance sheets/P&L and restatement audit notes are `financial_statements`.
- **Litigation inside General Info:** If a section under *General Information* details pending lawsuits, label it `litigation`.
- **Business vs. Industry:** What *the company itself* produces and operates is `business_operations`. Macro trends, sector statistics, and third-party industry reports are `industry_information`.
- **Objects vs. Capital Structure:** Where money *will go* is `objects_of_issue`. Current shareholding breakdown and past issuance history is `capital_structure`.

---

## 4. Special Reviewer Labels
- **`unsure`**: Use when the section covers multiple topics equally or text is genuinely ambiguous. **Do not guess.** Always pair `unsure` with a brief note in `reviewer_comment`.
- **`segmentation_error`**: Use in Files B, C, D, E, F, G, H when the text is corrupt, chopped in half, or is a table fragment missing its context due to PDF extraction failure.
- **`mixed_chapter`**: Use in File A (`01_chapter_mapping_review`) when a chapter title inherently spans multiple unrelated categories (e.g., *General Information* or *Statutory and Other Disclosures*) where a single global rule cannot apply.

---

## 5. Reviewer Workflow
1. Select `reviewer_label` from the dropdown list.
2. Select `reviewer_confidence`: `high` (clear-cut), `medium` (some nuance/minor overlap), or `low` (marginal decision).
3. Leave brief context in `reviewer_comment` whenever choosing `unsure`, `segmentation_error`, or `mixed_chapter`.

