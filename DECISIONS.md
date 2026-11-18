
# Decisions and Assumptions

This document records the assumptions and processing decisions used in
the Cantill Spend Intelligence pipeline.

## 1. Company A: Amounts and VAT

1. ERP `value_zar` is treated as net spend in ZAR.
2. Invoice-tool `gross_amount` is treated as a gross amount.
3. VAT is removed using:

   `net_amount = gross_amount / (1 + VAT rate)`

4. If the VAT rate is blank, 15% is used.
5. VAT rates greater than 1 are interpreted as percentages. For example,
   15 is interpreted as 15%.

## 2. Company A: Invoice Status

1. Only Approved and Paid invoice-tool records are eligible for inclusion.
2. Cancelled invoices receive the `CANCELLED` exclusion code.
3. Other non-approved statuses receive `NOT_APPROVED`.
4. ERP goods-receipt rows do not have the same invoice status field and
   are treated as valid receipts unless other exclusion rules apply.

## 3. Company A: Cross-System Matching

1. Purchase-order references are normalized by removing non-alphanumeric
   characters and converting text to uppercase.
2. Cross-system matching requires a non-empty normalized PO reference.
3. Matched records must also have net amounts within a 3% tolerance.
4. One ERP row can match only one invoice-tool row.
5. When a match is found, the ERP record is retained and the tool record
   is excluded as `DUPLICATE`.
6. Repeated invoice-tool exports are identified using normalized invoice
   number, supplier key, and net amount. The first eligible occurrence
   is retained.

This is a conservative matching strategy intended to reduce false-positive
matches.

## 4. Company B: Currency Conversion

1. Company B reports spend in USD.
2. The currency rate represents units of local currency per 1 USD.
3. Conversion is performed as:

   `USD amount = local currency amount / currency rate`

4. A valid row-level currency rate is preferred.
5. If the row-level rate is missing, the monthly FX file is used.
6. The invoice month is preferred; posting month is used as a fallback
   where available.
7. USD records are already in the reporting currency.
8. A non-USD amount without a resolvable rate is excluded rather than
   assigning an estimated or fabricated amount.

## 5. Company B: Labels and Duplicates

1. Exact seed-label decisions override heuristic decisions for the
   corresponding line IDs.
2. Seed labels are treated as useful but potentially noisy.
3. Exact duplicate invoice-line records are excluded.
4. Opposite-sign records are retained because they may represent valid
   reversals or credit notes.
5. Document type `CA` is treated as a cancellation.

## 6. Exclusion Rules

The pipeline uses the exclusion codes specified in the assignment brief:

| Code | Decision |
|---|---|
| CANCELLED | Cancelled invoice or line |
| NOT_APPROVED | Rejected, on hold, submitted, or disputed |
| NON_PROCUREMENT | Not third-party procurement spend |
| DUPLICATE | Already counted in another record |

Excluded records remain in the output with `include = 0` and `spend = 0`.

## 7. Supplier Keys

1. Normalized vendor identifiers are preferred when available.
2. Company A invoice-tool records may use supplier names where IDs are
   unavailable.
3. Supplier-name matching to ERP records is conservative.
4. Stable hashes are used to create repeatable keys from normalized values.

## 8. Item Keys

1. Material numbers are preferred when available.
2. If a material number is missing, a normalized item description is used.
3. The approach provides consistent keys but does not guarantee complete
   product-master matching.

## 9. Taxonomy and Classification

1. Taxonomy files are loaded at runtime.
2. Company-specific category names are not hardcoded.
3. Company A labels are not used to train its zero-label classifier.
4. Company B uses the supplied seed labels for supervised classification.
5. Classification confidence is based on maximum class probability or
   taxonomy-path similarity.
6. A taxonomy similarity score should not be interpreted as a calibrated
   probability.

## 10. Company C Blind Test

1. Company C is expected to contain an invoice-line file and an unseen
   taxonomy.
2. Reporting currency is USD, as specified in the assignment.
3. No labelled training data is assumed.
4. The YAML configuration maps Company C source columns to the fields
   expected by the generic engine.
5. The actual column names must be checked against the blind-test files.
6. No claim of successful Company C execution is made before those files
   are received and tested.

## 11. Data Privacy and Reproducibility

1. Raw assignment data is confidential and must not be committed to the
   public repository.
2. Configurations, code, documentation, tests, and permitted submission
   outputs can be committed.
3. The pipeline should be reproducible using the documented Python
   dependencies and company configuration files.
