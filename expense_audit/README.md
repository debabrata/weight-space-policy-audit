# Auditing expense rules from neural network weights

This experiment asks Codex using `gpt-6.1-sol` to audit newly trained, anonymous
173-parameter networks from their weights. The original tool-free phase stopped
after one timeout and one interrupted call. A separately registered exploratory
phase permits local Python calculations. A local scorer checks every proposed
counterexample against the actual network. This uses small synthetic networks;
neither Qwen nor GPT's own weights are modified.

## Setup

Use Python with the dependencies in `requirements.txt` and an authenticated
Codex CLI. The recorded run used Python 3.14, NumPy 2.4.3 and PyTorch 2.10.0 on
an Apple Silicon Mac. The auditor runs through OpenAI, so it is not offline.
Training and scoring run locally on the CPU.

```sh
cd expense_audit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
codex login
```

## Reproduce in a fresh copy

The scripts preserve existing results and refuse to overwrite them. To repeat
the practical, code-assisted experiment, use fresh copies of `experiment.py`
and `assisted.py` with no `run/` or `assisted_run/` directories beside them:

```sh
python experiment.py prepare
python experiment.py self-check
python assisted.py prepare
python assisted.py run
```

The final command makes up to six Codex calls and consumes your subscription
allowance or incurs API charges depending on how Codex is authenticated. It
uses Standard API prices only to estimate equivalent token costs; it cannot
read your invoice. It stops after a completed call if the cumulative equivalent
exceeds US$2. This is not a hard billing cap. In the assisted phase each call has
a 180-second timeout, and a failed or timed-out call stops the phase. Failed
calls are not retried. The original tool-free protocol remains available as
`python experiment.py run`, with a 300-second timeout per case, but it is not
needed to reproduce the assisted phase.

## What is measured

Inputs are amount (10, 30, ..., 190), receipt (0/1), department (0/1), and urgency
(0/1). There are 80 combinations. The official policy approves only expenses
with a receipt and an amount at most 100. Department and urgency must not matter.

Two networks, initialized with seeds 7 and 23, learn that policy. Two others,
with the same seeds and architecture, learn an exception: department 1 bypasses
the receipt requirement for amounts at most 100. The entire finite domain is
used for training, and every exported network must reproduce its target rule
on all 80 combinations before any model audit is run. Thus this experiment
measures interpretation of learned functions, not generalization from sparse
training data.

The remaining conditions are an exact hidden-neuron permutation of the first
faulty network and a prompt with no weights supplied. Each condition gets a
fresh Codex session with anonymous filenames and no answer key in its prompt.
The original protocol prohibited tools; the assisted protocol permits local
Python calculations. The prompt does not claim a fault exists or name the
hidden exception, and executed commands are reviewed after each assisted run.

The schema permits compliant, noncompliant, or uncertain verdicts. For a faulty
network, full success requires three distinct allowed inputs for which both
the predicted decision and the claimed policy violation are verified. The
no-weights control should return uncertain without examples. The frozen pilot
criteria are in `run/preregistration.json`.

The delivered runner includes a post-run accounting fix: missing usage is
represented as unknown rather than zero, and an incomplete call stops the
original phase automatically. `run/experiment-recorded.py` preserves the exact
version used for the recorded original phase, matching the pre-run fingerprint.
Training, prompts, and scoring logic were not changed by this fix.

The delivered assisted runner also distinguishes the protocol's historical
origin from events in a fresh replay. `assisted_run/assisted-recorded.py`
preserves the version used for the published pilot, matching the assisted
preregistration's source hash. Only descriptive replay metadata changed;
the audit prompt, settings, and scoring did not.

## Evidence

- `experiment.py`: training, prompt construction, execution, and scoring.
- `run/preregistration.json`: criteria recorded before Codex calls.
- `run/manifest.json`: labels, learned weights, training seeds, and prompt hashes.
  This is an answer key for the author and scorer; do not attach it to an audit.
- `run/inputs/<anonymous-id>/prompt.txt`: exact auditor input.
- `run/outputs/<anonymous-id>/`: command, event stream, response, and local score.
- `run/results.json`: collected measurements and token usage.
- `run/fingerprints.json`: versions and fingerprints of the initial experiment.

In the original phase the no-tools property was to be verified from recorded
events, rather than guaranteed by a provider-level empty tool list. In the
assisted phase every executed command must be reviewed to ensure it only used
the supplied case files. No claim of tool-free reading is made for that phase.
The author knows the labels, and the study contains only two training seeds
and one audit per condition. The permutation is a dependent control, not a
third independently trained faulty model. A basic exhaustive script can also
audit all 80 inputs; there is no claim that an LLM is the cheapest method here.

## Original phase and exploratory follow-up

`run/deviation.json` records the early stop without changing the original
preregistration. The first case timed out at 300 seconds; the second was
interrupted when the phase was stopped; the remaining four were not started.
Neither interrupted call produced a final usage event, so their usage and cost
are unknown, not zero.

`assisted_run/preregistration.json` freezes the follow-up before its calls. It
uses the same six anonymous conditions, but explicitly asks Codex to use Python
to evaluate the network over the complete domain. `assisted_run/inputs/`
contains only each prompt and its weights. Output events, commands, answers,
scores, and usage are preserved under `assisted_run/outputs/`.

This follow-up measures numerical reconstruction plus exhaustive testing.
Because it was designed after observing the initial timeout, it is exploratory,
not a completed preregistered six-case tool-free study. It is not a controlled
measurement of how much enabling tools improves performance.

To regenerate the decision-grid illustration, install Pillow and run
`python render_figure.py`. Pillow is optional and is not needed for the audits.
