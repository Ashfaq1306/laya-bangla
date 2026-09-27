# Bangla typed-decision smoke set v1

`bangla_smoke.jsonl` contains **36 synthetic support messages**: 12 scenarios,
each written in Bangla script, romanized Bangla (Banglish), and mixed
Bangla-English. It is original AI-authored project material, distributed under
the repository's Apache 2.0 license. No customer data or external dataset was used.

The text and labels are provisional and have not received independent fluent
speaker review. This is a diagnostic set for verifying the local pipeline and
spotting obvious failures, **not a representative benchmark or training set**.
Translations of the same scenario are correlated; there are 12 scenario groups,
not 36 independent observations. Do not use these results as a release threshold.

## Record format

Each UTF-8 JSONL object has:

- `id`: unique example ID.
- `group_id`: shared ID for equivalent scenarios/translations. Keep a group
  together when constructing any future splits.
- `slice`: `bangla`, `banglish`, or `mixed`.
- `state`: the original input string, preserving its spelling and Unicode.
- `expected`: one label per question ID in the versioned question schemas.

## Tasks and labeling rubric

| Question | Type | Labels |
| --- | --- | --- |
| `department` | `choice` | `billing`, `technical`, `sales`, `other` |
| `churn_risk` | `noul` | Boolean: an explicit, possibly conditional threat to leave |
| `urgency` | `score` | Integer 0–2: no urgency, soon/tomorrow, now/today/blocked |

Route by the main subject. Delivery and opening-hours questions belong to
`other`. A complaint alone does not establish churn risk. Explicitly denying
an intention to leave is a negative case. Use the stated deadline and work
blockage for urgency, rather than guessing from the department or tone.

Each input slice has 3 cases per department, 6 churn-positive and 6 churn-negative
cases, and 4 cases per urgency level. The fixed majority baselines are therefore
25%, 50%, and 33.3%, respectively. These balanced counts are designed for a
smoke check and do not reflect real customer traffic.

`configs/questions.bn.json` and `configs/questions.en.json` express the same
tasks and ordered labels with Bangla and English instructions/descriptions.
The runner evaluates the same checkpoint with both schemas. These are prompt
variants, not additional independent examples.

## Growing the evaluation

Keep this small diagnostic set versioned. Build a separately sourced,
speaker-reviewed dataset with group-aware train, validation/calibration, and
test splits before fine-tuning. Document source licenses, label disagreements,
class frequencies, and the workflow-specific evaluation protocol. Preserve
negation and numerals during normalization, and keep raw text for error review.
