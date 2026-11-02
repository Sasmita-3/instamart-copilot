# Instamart Customer Intelligence & Support Copilot

## Overview

A reproducible, offline-first capstone built from deterministic synthetic data. The repository connects a cleaning and SQLite pipeline (Part A), RFM customer segmentation and lapse prediction (Part B), privacy-aware support triage (Part C), and TF-IDF policy retrieval plus a tool-using refund agent (Part D). All monetary values are in INR. The supplied data generator and mock LLM are included unmodified; no network model or API key is used.

## Reproduce

Run these commands from the repository root in PowerShell or a POSIX shell (use the equivalent activation command for your shell). Python 3.11+ is required.

1. `git init -b main` (when starting a new local repository; for a GitHub clone, stay on `main`).
2. `python -m venv .venv`
3. Windows PowerShell: `.\.venv\Scripts\Activate.ps1` (POSIX: `source .venv/bin/activate`)
4. `python -m pip install -r requirements.txt`
5. `python -m pip freeze` (inspect the installed environment; keep the repository’s tested pinned requirements unchanged)
6. `python generate_data.py`
7. `python src/part_a.py`
8. `python src/validate.py`
9. `python src/part_b.py`
10. `python src/part_c.py`
11. `python src/part_d.py`
12. `python -m unittest discover -s tests -v`

Alternatively, `python run_all.py` runs the generator and all four parts in order. The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Fixed random seeds are used for synthetic data, clustering, splitting, model tuning and oversampling. The model is local and offline.

## Part A

### Raw-data audit

`outputs/raw_audit.csv` records table shape, dtype, missing count and missing percentage for each column. `outputs/audit_checkpoints.csv` records the specific audit figures. Expected raw audit checkpoints from the brief: 13,084 order rows; 119 exact duplicate rows; 656 pending rows; 12,309 distinct order IDs; 120 unparseable amounts; 106 negative delivery durations; 2,541 blank delivery durations; 61 blank ages; 6 ages outside 10–100; 24 raw city spellings; 10 nonblank gender spellings; 30 blank genders.

### Cleaning rules

- Customers: strip city whitespace and title-case values; map gender spellings to `male`, `female`, or `unknown`; set ages outside 10–100 to missing and impute missing ages with the median of the cleaned city using `groupby(...).transform`; parse signup dates.
- Orders: parse timestamps; remove exact duplicate rows; sort by timestamp and retain the latest record per order ID; parse amounts and calculate missing amounts from catalogue price × quantity × (1 − discount/100); set negative delivery durations missing; impute missing duration only for completed orders using the completed-order median for the product category. Cancelled orders remain without a delivery duration.
- Products: preserve the supplied catalogue values and numeric price field.
- Clean CSVs are written to `data/clean/`; the same three tables are stored in `data/instamart.db`.

The validation script asserts the key customer/order count, status, missingness and completed-revenue checkpoints and prints `ALL CHECKPOINTS PASSED` only if all pass. Seven SQL queries are stored in `sql/queries.sql`; result CSVs are emitted in `outputs/`. Query 6 uses `RANK`, `DENSE_RANK`, and `ROW_NUMBER`; ties cause `RANK` to skip rank numbers while `ROW_NUMBER` always assigns a unique sequential position. EDA and pandas cross-checks are in `outputs/pandas_revenue_pivot.csv`, `outputs/city_spend_stats.csv`, and `outputs/eda_summary.json`.

The completed-order spend distribution is right-skewed: high-value electronics purchases pull the mean above the median. The 1.5×IQR fence identifies high-spend outliers; these are retained rather than removed because they can represent genuine purchases. Correlations describe linear association, not causation; the largest absolute correlation among the requested fields is expected to be order amount versus discount percentage, and a correlation of this small magnitude does not establish causation. Exact run-specific statistics are in `outputs/eda_summary.json`.

## Part B

The RFM table is calculated at 2025-10-01 00:00:00 from completed orders. The three RFM fields are standardized before K-means; K=2…8 inertia and silhouette values are saved to `outputs/kmeans_scores.csv`. K=4 is selected as a balance of the elbow, silhouette score and actionable segments. Inertia is lower for K=8 because more clusters can fit the observations more closely; lower inertia alone always rewards additional clusters and does not establish that the segments are useful.

`outputs/segment_profiles.csv` contains sizes and mean RFM values; `outputs/customer_segments.csv` includes the segment ID, business name and Premium flag. Cluster numbers are arbitrary identifiers, not an ordinal ranking. Premium is assigned to the segment with the highest mean monetary value. Recommended actions: Premium—VIP retention and proactive service; Frequent Essentials—replenishment reminders and bundles; Occasional Shoppers—cross-category discovery offers; Dormant Low-Spend—low-cost win-back incentives. The two-component PCA variance ratios are in `outputs/part_b_report.json`.

The lapse cohort is defined by at least one completed order before 2025-09-01; lapse means no completed order during September 2025. All predictive order features are built strictly from pre-cutoff orders. The nominal encoder and numeric scaler are fitted inside a pipeline on training rows only; test rows are only transformed. Segment labels are not model features. The report includes a majority-class baseline, logistic-regression and random-forest grid searches, held-out metrics/confusion matrices, a training-only oversampling comparison, random-forest feature importances, logistic coefficient signs, model reload equality and the retention list.

The retention list is illustrative and not a leakage-free evaluation because the Premium flag is derived from whole-window RFM, which includes the September label window; the predictive model's own features use only pre-cutoff data.

## Part C

Tickets are processed in file order. Empty text is skipped and text estimated above 400 tokens is flagged before any model call. Email, phone and order identifiers are redacted in the required order, and assertions check that none remain before a prompt is sent. The user message contains only the redacted ticket text. Each model call has exponential-backoff retries with jitter and a retry log. Replies are accepted only as allowed-category JSON or when exactly one allowed category is extracted with `re.findall(r"[a-z]+", ...)`; invalid outputs go to human review rather than guessed categories.

The zero-shot and few-shot run summaries, confusion matrix, token use and redaction counts are in `outputs/part_c_report.json`; individual retry messages are also saved in `outputs/triage_retry_log.txt`. `outputs/triage_budget_demo.csv` demonstrates budget exhaustion; `outputs/triage_results.csv` has all 164 tickets routed according to status, fraud priority and Premium membership. Few-shot prompts cost more tokens but reduce the mock classifier's systematic confusion of fraud/technical reports as billing; use the measured token/accuracy comparison in the report to decide whether the extra prompt cost is justified for this workload.

## Part D

The five policy documents are split into 40-word chunks with a 10-word overlap. A fitted `TfidfVectorizer` ranks chunks by cosine similarity. TF measures within-document term frequency; IDF downweights terms common across documents. TF-IDF can miss semantically equivalent paraphrases when they use different vocabulary. The policy assistant uses the three retrieved chunks as context and returns a fallback beginning `I could not find` when the mock cannot find an answer in context. `outputs/policy_eval_results.csv` and `outputs/policy_eval_no_overlap.csv` compare the overlapping and zero-overlap runs.

The refund agent has five named tools in `TOOLS_SPEC` and a corresponding `TOOL_MAP`. It reads orders from SQLite, retrieves policy chunks, computes eligibility, and either records an automatic refund or escalates amounts above INR 1000. The simulated ledger is not a real payment system. The issue/escalation tools independently verify order completion, delivery time, exact refund amount, and the INR 1000 approval threshold; the synthetic scenario treats all requests as within seven days. Every tool observation is checkpointed in `state/`; `resume_agent` continues from saved messages and does not repeat completed tools. `outputs/agent_runs.jsonl` contains the seven scenario runs, and `outputs/refund_ledger.jsonl` records simulated issue/escalation actions.

The seven scenario results and run-specific token/call counts are recorded in `outputs/part_d_report.json`. The S1 token-limit demonstration must return `FALLBACK_TOKEN_LIMIT`; the zero-second timeout demonstration must return `FALLBACK_TIMEOUT`; S7 must end at `FALLBACK_MAX_ITERATIONS`. The S1 crash/resume sequence should execute `get_order`, `search_policy`, `calculate_refund`, and `issue_refund` exactly once each, in that order, and finish with `REFUND` for INR 128.25.

ReAct makes sequential model calls after observing each tool result, which costs more calls/tokens than a plan-and-execute design that proposes a whole tool sequence once. However, ReAct can change course after a cancelled order, missing order or zero-refund calculation; a preplanned sequence can issue inappropriate follow-up actions if it does not validate each observation. For refunds, I prefer ReAct with strict deterministic tools and limits because the cost of an incorrect financial action is higher than the additional offline-model calls.

## Limitations and Next Steps

- The dataset and all policies are synthetic and are not real company records or legal commitments.
- TF-IDF uses lexical overlap rather than semantic embeddings; retrieval quality can fall on paraphrases.
- The mock LLM is deterministic and designed to exercise retry, validation and agent-control logic, not to represent production model quality.
- The refund ledger only records simulated actions; it does not move money.
- The retention list is illustrative and not a leakage-free evaluation because Premium uses whole-window RFM including the label window. A production evaluation should define segment membership at the prediction cutoff.
- Before production use, add human review, monitoring, audit retention controls, stronger authorization, real policy versioning and an independently held-out time-based evaluation.

## Verified run results and reference checkpoints

The following tables are generated from the checked run in this repository. Full machine-readable metrics remain in the `outputs/` JSON and CSV files.

### Part A — audit, cleaning and SQL results

**Raw table column audit.** Rows/columns are repeated on each column row so each field is self-contained. Blank strings are counted as missing for this audit.

| Table | Shape (rows × columns) | Column | Raw dtype | Missing count | Missing % |
|---|---:|---|---|---:|---:|
| customers | 1200 × 5 | customer_id | object | 0 | 0.00% |
| customers | 1200 × 5 | city | object | 0 | 0.00% |
| customers | 1200 × 5 | signup_date | object | 0 | 0.00% |
| customers | 1200 × 5 | age | object | 61 | 5.08% |
| customers | 1200 × 5 | gender | object | 30 | 2.50% |
| products | 40 × 4 | product_id | object | 0 | 0.00% |
| products | 40 × 4 | product_name | object | 0 | 0.00% |
| products | 40 × 4 | category | object | 0 | 0.00% |
| products | 40 × 4 | price_inr | int64 | 0 | 0.00% |
| orders | 13084 × 9 | order_id | object | 0 | 0.00% |
| orders | 13084 × 9 | customer_id | object | 0 | 0.00% |
| orders | 13084 × 9 | product_id | object | 0 | 0.00% |
| orders | 13084 × 9 | quantity | int64 | 0 | 0.00% |
| orders | 13084 × 9 | order_ts | object | 0 | 0.00% |
| orders | 13084 × 9 | status | object | 0 | 0.00% |
| orders | 13084 × 9 | discount_pct | int64 | 0 | 0.00% |
| orders | 13084 × 9 | order_amount | object | 0 | 0.00% |
| orders | 13084 × 9 | delivery_minutes | object | 2541 | 19.42% |
| tickets | 164 × 5 | ticket_id | object | 0 | 0.00% |
| tickets | 164 × 5 | customer_id | object | 0 | 0.00% |
| tickets | 164 × 5 | order_id | object | 8 | 4.88% |
| tickets | 164 × 5 | text | object | 4 | 2.44% |
| tickets | 164 × 5 | true_category | object | 0 | 0.00% |

**Raw audit checkpoints**

| Metric | Observed |
|---|---:|
| raw orders rows | 13084 |
| exact duplicate rows | 119 |
| pending rows | 656 |
| distinct order ids | 12309 |
| non numeric amounts | 120 |
| negative delivery minutes | 106 |
| blank delivery minutes | 2541 |
| blank age | 61 |
| age outside 10 100 | 6 |
| raw city spellings | 24 |
| raw nonblank gender spellings | 10 |
| blank gender | 30 |

**Clean customer counts**

Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender: female 610, male 560, unknown 30. Ages missing after imputation: 0.

**Clean orders:** 12,309 unique orders; 11,110 completed and 1,199 cancelled; pending rows 0; missing completed delivery times 0; missing cancelled delivery times 1199; completed revenue INR 2,998,668.80.

**Seven SQL query outputs**

| Query | Returned rows | Required result/checkpoint |
|---|---:|---|
| Q1 — WHERE/HAVING | 30 | 30 customers have completed orders above INR 500 with grouped spend above INR 8,000. |
| Q2 — city × category join | 30 | Grand total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45. |
| Q3 — above-average catalogue price | 11 | 11 products. |
| Q4 — LEFT JOIN without completed orders | 25 | 25 customers. |
| Q5 — CTE above-average spend | 550 | Mean customer completed spend INR 2,552.06. |
| Q6 — rank functions | 1,175 | Eight rows have RANK different from ROW_NUMBER because ties share a rank while ROW_NUMBER assigns each row a unique position. |
| Q7 — daily running revenue | 183 | Final cumulative revenue INR 2,998,668.80. |

**Q6 top spender by city**

| City | Customer | Spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

**Completed-order spend by city**

| City | Sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**Completed-order EDA:** mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; lower fence INR -283.50; upper fence INR 645.30; outliers below/above fences 0/897. Largest absolute requested correlation: order_amount vs discount_pct = -0.095; order_amount vs quantity = 0.087. A correlation of this size does not establish causation.

### Part B — clustering and lapse model

**RFM checkpoint:** 1175 customers; recency median/min/max 19/0/180 days; frequency mean/max 9.46/33; mean/max monetary INR 2552.06/14766.55.

**K-means selection table**

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K=4 segment profiles**

| Segment ID | Business name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1057.62 | No |
| 1 | Premium | 138 | 24.12 | 6.77 | 7129.70 | Yes |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | 3410.02 | No |
| 3 | Dormant Low-Spend | 182 | 109.04 | 2.20 | 685.99 | No |

PCA explained variance ratios: 0.634, 0.218 (total 0.852). Cohort 1156; lapsed 443 (38.32%); train/test 924/232. Baseline accuracy 0.616; lapsed F1 0.000.

**Tuned model evaluation on the untouched test set**

| Model | Best parameters | Best CV F1 | Confusion matrix (true rows 0,1; predicted columns 0,1) | Precision (lapsed) | Recall (lapsed) | F1 (lapsed) | Accuracy |
|---|---|---:|---|---:|---:|---:|---:|
| Logistic regression | `{'model__C': 10}` | 0.656 | `[[104, 39], [32, 57]]` | 0.594 | 0.640 | 0.616 | 0.694 |
| Random forest | `{'model__max_depth': 8, 'model__min_samples_leaf': 10}` | 0.683 | `[[113, 30], [32, 57]]` | 0.655 | 0.640 | 0.648 | 0.733 |

The oversampled training split is balanced to 570 lapsed and 570 non-lapsed rows by duplicating lapsed training rows only. Training-only oversampled logistic regression: precision 0.595, recall 0.876, F1 0.709, confusion matrix `[[90, 53], [11, 78]]`. Recall moves most, rising from 0.640 to 0.876; that catches more potential lapsers but increases false-positive outreach. Training scaler frequency mean 8.115; full-cohort mean 8.029; they differ because the scaler is fitted only on training rows.

Top five random-forest importances (monetary and frequency dominate, so historical spend and repeat ordering are the strongest signals; recency, delivery time and cancellations add secondary information):

| Feature | Importance |
|---|---:|
| num__monetary | 0.282 |
| num__frequency | 0.274 |
| num__recency | 0.118 |
| num__avg_delivery | 0.059 |
| num__cancellation_rate | 0.054 |

Three largest absolute logistic coefficient examples (a negative coefficient means the feature lowers the log-odds of lapse relative to the encoded reference category or per scaled unit):

| Feature | Coefficient | Sign |
|---|---:|---|
| num__frequency | -1.589 | negative |
| cat__favourite_category_Electronics | -0.810 | negative |
| cat__city_Delhi | -0.352 | negative |

Model reload reproduced test predictions exactly: True. Retention list rows: 10; truly lapsed in the list: 1.

### Part C — ticket triage results

**PII redaction counts**

| Type | Substitutions |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| **Total** | **278** |

Redaction matches remaining after rechecking: 0. Empty tickets skipped: 4; long tickets flagged before a call: 4.

**Zero-shot versus few-shot**

| Run | Classified | Correct | Accuracy | Total tokens | Tokens/classified ticket | JSON | Keyword | Invalid output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.7% | 11,428 | 77.74 | 135 | 12 | 9 |
| Few-shot | 147 | 147 | 100.0% | 18,604 | 126.56 | 135 | 12 | 9 |

**Zero-shot confusion matrix (actual rows, predicted columns)**

| Actual \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

At the 6000 token budget, 74 tickets were marked budget_exhausted. The final few-shot pipeline routed all 164 ticket rows. Few-shot used 48.82 additional tokens per classified ticket and improved observed accuracy by 14.3 percentage points (85.7% to 100.0%); this is worth considering for fraud/technical tickets because the zero-shot run misrouted 11 fraud and 10 technical tickets as billing.

### Part D — policy retrieval and refund agent

**Chunk counts**

| Policy file | Chunks |
|---|---:|
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| refund_policy.txt | 4 |
| **Total** | **15** |

**Policy evaluation**

| Chunking | Expected document in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Unanswerable fallback (2) |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the question “What delivery time makes a completed order eligible for a full refund?” fails the expected-fact check because the phrase describing the threshold is split across a chunk boundary; overlapping chunks preserve the complete phrase in retrieved context.

**Refund agent scenario results**

| Scenario | Status | Expected refund (INR) | Final answer | Model calls | Tokens |
|---|---|---:|---|---:|---:|
| S1 | REFUND | 128.25 | REFUND: INR 128.25 issued. | 5 | 937 |
| S2 | REFUND | 135.00 | REFUND: INR 135.0 issued. | 5 | 937 |
| S3 | ESCALATE | 1999.00 | ESCALATE: refund above INR 1000 sent to a human lead. | 5 | 924 |
| S4 | DENIED | — | DENIED: delivery was within 45 minutes. | 4 | 637 |
| S5 | DENIED | — | DENIED: only completed orders can be refunded. | 3 | 361 |
| S6 | NOT_FOUND | — | NOT_FOUND: no such order. | 2 | 88 |
| S7 | FALLBACK_MAX_ITERATIONS | — | FALLBACK_MAX_ITERATIONS: Hand off to a human agent; maximum model iterations reached. | 6 | 3048 |

Stop-safely demonstrations: token limit → **FALLBACK_TOKEN_LIMIT**; zero-second timeout → **FALLBACK_TIMEOUT**. Crash/resume final status: **REFUND**; tool sequence across crash and resume: `get_order → search_policy → calculate_refund → issue_refund`. The checkpoint is written after each observation so already-completed tools are not repeated.
