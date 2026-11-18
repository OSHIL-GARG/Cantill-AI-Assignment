
# Approach

## 1. Solution Overview

The solution is a configuration-driven Python pipeline designed to process
procurement and accounts-payable data for multiple companies. Each company
has a separate YAML configuration containing its input files, taxonomy,
reporting currency, and output path.

The pipeline is organized into ingestion, preprocessing, integration,
deduplication, spend calculation, classification, evaluation, and output
generation stages.

## 2. Company A

Company A contains ERP goods-receipt records and invoice-tool records.
ERP values are treated as net ZAR. Invoice-tool gross amounts are converted
to net values by removing VAT. A missing VAT rate defaults to 15%.

Approved and Paid invoice-tool records are eligible for inclusion.
Cross-system matching uses normalized purchase-order references and a
net-amount tolerance. When a tool record matches an ERP record, the ERP
record is retained and the tool record is excluded as DUPLICATE.

Repeated invoice-tool exports are identified using invoice number,
supplier key, and net amount.

Company A classification uses the taxonomy supplied at runtime.
The current zero-label baseline compares purchase descriptions with
taxonomy paths using TF-IDF-style text similarity. Company A labels are
not used to train this classifier.

## 3. Company B

Company B contains invoice lines from multiple ERP systems and currencies.
Amounts are converted to USD using the row-level currency rate when
available. If the rate is missing, the monthly FX file is used.
The FX rate is interpreted as units of local currency per 1 USD.

A TF-IDF and Logistic Regression classifier is trained using the supplied
seed labels. The taxonomy is loaded at runtime. Exact seed decisions are
preserved for matching line IDs.

Non-procurement and non-approved records are excluded using status,
account-description, and text-based rules. Exact duplicate invoice-line
records are excluded while opposite-sign records are retained as possible
reversals or credit notes.

## 4. Supplier and Item Matching

Supplier keys are generated using normalized supplier identifiers where
available. For Company A invoice-tool records, supplier names are used
when identifiers are unavailable; matching to ERP suppliers is
conservative.

Item keys use material numbers when available. Otherwise, normalized
item descriptions are used to generate stable keys.

These methods provide consistent identifiers but do not guarantee that
every spelling variation or equivalent product is merged correctly.

## 5. Classification and Confidence

Taxonomy category names are read from the company's taxonomy file at
runtime. The classification output follows the available hierarchy and
does not hardcode company-specific category names.

Confidence is based on the classifier's maximum class probability or
taxonomy-path similarity score. The Company A similarity score is not a
calibrated probability.

## 6. Measured Results

The following results are recorded from the local task-kit run:

| Measure | Result |
|---|---:|
| Company A output rows | 65,549 |
| Company A runtime | Approximately 9.5 seconds |
| Company B output rows | 41,028 |
| Company B runtime | Approximately 16.9 seconds |
| Company A L1 accuracy, labels hidden | 29.24% |
| Company A L2 accuracy, labels hidden | 11.51% |
| Company A L3 accuracy, labels hidden | 8.98% |
| Company B held-out leaf accuracy | 93.73% |
| Company B held-out leaf macro-F1 | 82.73% |
| Unit tests | 3 passed |

Company A's zero-label classification is a known weakness. Its taxonomy
similarity baseline has limited ability to interpret abbreviations and
short or ambiguous descriptions.

Company B's held-out evaluation used 2,038 training rows and 510 test rows.
Two singleton target labels were excluded from stratified holdout.

Company C cannot be empirically evaluated until its blind-test dataset
and exact column names are supplied.

## 7. Runtime and External API Cost

The recorded local runtimes are well below the assignment's 60-minute
execution limit.

The implementation uses local Python libraries and does not require a
paid LLM or external API. Therefore, external API cost is zero.

## 8. Limitations

- Company A's zero-label classifier requires improvement.
- Fuzzy supplier and item matching is conservative.
- Missing or reformatted purchase-order references can reduce matching.
- Company B seed labels may contain noise.
- Company C's generic pipeline requires validation against its actual
  blind-test file structure.
