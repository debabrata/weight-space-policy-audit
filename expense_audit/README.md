# Reproduce the Sol and Astra policy audit

The [article](ARTICLE.md) explains the experiment. The [results](comparison_run/RESULTS.md) report the completed tool-free runs: both models passed the faulty, compliant and missing-weights cases, plus a separate format-only check. Each faulty-case answer contained three verified counterexamples in the required field.

## Environment

The recorded environment used Python 3.14, NumPy 2.4.3, PyTorch 2.10.0, jsonschema 4.26.0 and Codex CLI 0.164.0 on an Apple Silicon Mac. Training and scoring run locally on CPU. Auditors run through OpenAI, so the complete experiment is not offline. Fresh calls require Codex access to `gpt-6.1-sol` and `gpt-6-astra`.

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r expense_audit/evaluation-requirements.txt
```

## Verify saved evidence without model calls

```sh
python expense_audit/comparison_verify.py
python expense_audit/experiment.py self-check
```

The independent verifier parses parameters from the saved prompts and uses standard-library forward arithmetic on all 80 inputs. It does not import the training harness, NumPy, Torch or manifest labels. It checks verdicts, submitted examples, schemas, format copies and observed tool events. The separate scorer check exercises correct and incorrect examples.

## Run a fresh comparison

Existing evidence cannot be overwritten. Copy the code into a fresh directory:

```sh
mkdir expense-audit-replay
cp expense_audit/experiment.py expense_audit/comparison.py expense_audit/comparison_verify.py expense-audit-replay/
cd expense-audit-replay
python experiment.py prepare
python experiment.py self-check
python comparison.py prepare
```

These commands train/export the synthetic data pool, validate the conditional schema and scorer, and freeze the protocol. They make no model calls. The pool contains six checkpoints/controls; the comparison selects only `80283345` (faulty), `328a2eee` (compliant) and `b422dcb7` (missing weights). Unused data is not counted as an audit result.

The next commands consume subscription allowance or incur API charges:

```sh
codex login
python comparison.py sol
python comparison.py astra
python comparison_verify.py
```

Sol runs first. Each model receives a format-only preflight and three independent audits, using high reasoning. The preflight has a 60-second limit; audits have 300 seconds each. Task prompts and schema are identical; model-specific CLI overhead and token totals may differ. Auditors are instructed not to use tools; observed tool use is logged and terminates that arm.

There are at most eight calls, with no retries. Failed preflights skip that arm's audits. Wrong audit answers and timeouts remain recorded while the fixed pilot continues; provider/CLI errors terminate that arm. Known cumulative Standard API-equivalent usage above $10 stops further calls. This is not a hard invoice cap; missing usage and actual subscription charges remain unknown. No larger evaluation starts automatically.

## Policy and scoring

Inputs are amount in `{10,30,50,70,90,110,130,150,170,190}`, receipt, department and urgent (each binary): 80 combinations. Approve exactly when amount is at most 100 and a receipt exists. Department and urgency must not change the decision.

The network is `4 → 12 → 8 → 1`, with tanh hidden layers and 173 parameters. Input is `[amount/100, receipt, department, urgent]`; nonnegative logit means approve. Six-decimal exports define the evaluated network. All 80 inputs were used for training; there is no held-out-input generalization claim.

The faulty network was deliberately trained to let department 1 bypass receipts below the amount limit, producing ten violations. The auditor sees policy, architecture and weights, without checkpoint labels, training source or answers.

The schema wraps answers in `result`. A noncompliant verdict requires exactly three objects in `result.examples`; compliant/uncertain verdicts require an empty list. Schema validity, verdict correctness and verified examples are reported separately. Examples must be allowed, distinct, genuinely violate policy and correctly predict the network. Scoring does not automatically validate every sentence of the prose explanation.

## Limits and provenance

This is one attempt per model per case, using reused public checkpoints. Both models passed this small comparison. It does not establish general reliability, model superiority, an internal reasoning mechanism or weight repair. Numerical reasoning on inputs was allowed; this is not a protocol forbidding all forward calculations.

[Source provenance](comparison_run/source-provenance.json) retains the preparation/scoring source used at execution and explains the cleaned preparation module supplied for replay.
