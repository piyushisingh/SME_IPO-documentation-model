# Annotation guidelines — SME IPO section taxonomy (v4, provisional)

**Status: PROVISIONAL.** These guidelines describe the rules the *automated* labeler (`scripts/09_build_master.py`) applied. No human has verified any label. A human reviewer should use this guide to confirm or correct labels; `review_status` is never set to `verified` by any script.

Taxonomy = the existing 17 category names, unchanged: `ipo_details, risk_factors, disclosures_and_definitions, legal_and_regulatory, promoters_and_management, company_information, financial_information, financial_statements, related_party_transactions, capital_structure, objects_of_issue, qualitative_factors, industry_information, business_operations, litigation, declarations, other`.

## 1. Unit and primary-label rule
- **L1** = a chapter taken from the document's own table of contents. **L2** = a sub-section inside an L1 chapter (bold heading).
- **Primary-label rule:** one primary label per record. A chapter with a single dominant topic gives that label to the chapter and to all its L2 sub-sections (`label_basis = inherited_from_chapter_rule`). A sub-section that genuinely belongs elsewhere is relabeled only by an explicit heading-keyword rule (section 3). A multi-topic chapter gets **no** chapter label.
- Classification should follow **content and context, not heading alone**. The automated labeler cannot read content; the only content check is a weak lexical cue test (`suspected_label_content_mismatch`, density ≥ 4 cue hits per 1,000 characters, 7 cue categories). It is a prompt for human review, not verification.

## 2. Chapter-title rules (rule id → category)
| rule | chapter titles (normalised, any document) | category | overlap noted |
|---|---|---|---|
| C01 | Definitions and Abbreviations; Certain Conventions…; Presentation of Financial, Industry and Market Data; Forward-Looking Statements | disclosures_and_definitions | |
| C02 | Risk Factors | risk_factors | |
| C03 | The Issue / The Offer; Issue (Offer) Structure; Issue (Offer) Procedure; Terms of the Issue (Offer) | ipo_details | |
| C04 | Restrictions on Foreign Ownership of Indian Securities | legal_and_regulatory | ipo_details |
| C05 | Capital Structure | capital_structure | |
| C06 | Capitalisation Statement | financial_information | capital_structure |
| C07 | Objects of the Issue (Offer) | objects_of_issue | |
| C08 | Basis for Issue (Offer) Price | ipo_details | financial_information |
| C09 | Statement of (Possible) (Special) Tax Benefits | legal_and_regulatory | financial_information |
| C10 | Industry Overview; Our Industry | industry_information | |
| C11 | Our Business; Business Overview | business_operations | |
| C12 | Key (Industry/Industrial) Regulations and Policies | legal_and_regulatory | industry_information |
| C13 | (Our) History and Corporate Structure/Matters | company_information | |
| C14 | Our Management | promoters_and_management | |
| C15 | Our Promoter(s) and Promoter Group | promoters_and_management | |
| C16 | Our Group Company/Companies; Our Subsidiaries… | company_information | promoters_and_management |
| C17 | Summary of Related Party Transactions | related_party_transactions | |
| C18 | Dividend Policy | financial_information | capital_structure |
| C19 | Summary of (Our) Financial Information/Statements; Summary of Contingent Liabilities; Other Financial Information; (Statement of) Financial Indebtedness; Management's Discussion and Analysis | financial_information | |
| C20 | Restated (Audited) Financial Information/Statements; Statement of Assets…/Cash Flow/Profit… as Restated | financial_statements | |
| C21 | Outstanding Litigation(s) and Material Developments | litigation | |
| C22 | Government and Other (Statutory/Key) Approvals; Other Regulatory and Statutory Disclosures | legal_and_regulatory | |
| C23 | Material Contracts and Documents for Inspection; Main Provisions of the Articles of Association; Description of Equity Shares and Terms of the Articles | legal_and_regulatory | |
| C24 | Declaration | declarations | |
| C25 | General Information | *(none — multi-topic)* | |
| C26 | Abridged Prospectus (Salient Features) *(AXIOM pp. 1–13 only)* | *(none — multi-topic)* | |

Records labeled through an *overlap* row carry `label_confidence_tier = rule_chapter_overlap`, flag `category_overlap_possible`, and the alternative in `category_proposed`.

## 3. Heading-keyword rules (L2)
- **Overrides inside single-topic chapters:** K01 "related party" → `related_party_transactions` (only inside financial_statements / financial_information chapters); K02 "qualitative factors" → `qualitative_factors` (inside Basis for Issue Price); K03 "quantitative factors" → `financial_information` (inside Basis for Issue Price); K04 heading starts "Declaration" → `declarations`; K05 heading contains "litigation" → `litigation`. Overridden records carry `heading_keyword_overrides_chapter_label` and the original chapter label in `category_proposed`.
- **General Information (multi-topic):** G01 registered office / incorporation / brief summary → `company_information`; G02 board of directors / KMP / company secretary / compliance officer / CFO → `promoters_and_management`; G03 intermediaries and issue mechanics (credit rating, IPO grading, brokers, trustees, registrar, bankers, SCSBs, syndicate, depositories, monitoring agency, underwriting, market making, book building, green shoe, withdrawal, grievances, expert opinion, accountant, lead manager…) → `ipo_details`. Any other General Information sub-section → `ambiguous`.
- **Abridged prospectus (multi-topic):** A01–A07 map objects/utilisation, promoters, shareholding, litigation, declaration, business items and financial measures; anything else → `ambiguous`.

## 4. Overlapping categories — decisions made
- `financial_statements` = the restated statements themselves (balance sheet, P&L, cash flow, notes/annexures). `financial_information` = everything numeric about finances *outside* the statements (summary tables, MD&A, indebtedness, contingent liabilities, capitalisation, dividend policy, quantitative price factors).
- `capital_structure` = share capital, history of capital, shareholding patterns. The capitalisation statement (debt/equity table) is placed in `financial_information` and flagged.
- `legal_and_regulatory` = statutory/regulatory framework, approvals, tax-benefit statement, Articles, material contracts. `litigation` = specific proceedings, claims and amounts. Court/tribunal wording alone does **not** make a section `litigation`.
- `promoters_and_management` covers promoters, promoter group, directors, KMP. Group companies and subsidiaries go to `company_information` (flagged overlap).
- `declarations` = formal declarations/certifications by the company, directors, selling shareholders. SEBI disclaimer clauses are inside "Other Regulatory and Statutory Disclosures" and are therefore labeled `legal_and_regulatory`.
- `qualitative_factors` is used only by rule K02 (6 records in total). The taxonomy gives no other definition; reviewers should decide whether to keep it as a category.

## 5. Nested sections, tables, appendices, boilerplate
- L2 inherits from its L1 unless a keyword rule applies. Heading-only L2 records (a heading with no body) are merged into the **following** sibling when building model-ready data (backward if last in the chapter); provenance of every merged record is kept.
- Tables are kept as `pdftotext -layout` text (columns aligned with spaces). Rows/columns are **not** parsed. Financial tables inside `financial_statements`/`financial_information` should be spot-checked against the PDF before use.
- Definitions glossary, Articles of Association and similar long unbroken text are single long sections (not split).
- Contents pages and cover/front matter are structural (`excluded_structural`), never labeled, never in model-ready data.
- Repeated boilerplate (SEBI disclaimers, standard ICDR wording) is kept as legitimate disclosure; near-duplicates across documents are reported in `duplicate_report.csv`, not removed.

## 6. When to use `other`
`other` is **never assigned automatically**. A reviewer may assign it only when a section is genuine content that fits none of the 16 other categories. Currently 0 records are `other`.

## 7. When a record is ambiguous / needs review
- `ambiguous`: multi-topic chapter with no keyword hit (67 records at time of writing), or no chapter rule matched. Never forced into a label.
- `extraction_error`: text unreliable (Quanto scanned financial statements, pp. 236–271).
- `needs_review`: every rule-labeled record (all of them are unverified).
- Extra review flags: `suspected_label_content_mismatch`, `heading_keyword_overrides_chapter_label`, `fragment_heading_suspected` (e.g. a heading that is only "Herring Prospectus"), `heading_is_paragraph_suspected` (a whole bold paragraph detected as a heading, e.g. the SEBI disclaimer), `category_overlap_possible`.

## 8. Known limitations of this guide
It was written from chapter titles and sub-heading lists observed in six documents; it has not been tested by a second annotator and no inter-annotator agreement exists.
