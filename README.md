# Weight space policy audit

Can Codex infer a hidden expense-policy exception from a small neural network's
weights and demonstrate it with concrete inputs?

This repository contains an experiment with **Codex and GPT-6.1 Sol**, its article,
and the recorded evidence. Training and scoring run locally; the auditor uses
OpenAI through the Codex CLI.

**[Read the article](expense_audit/ARTICLE.md)** ·
**[Reproduction guide](expense_audit/README.md)** ·
**[Recorded results](expense_audit/assisted_run/results.json)**

![Actual decisions of compliant and faulty networks](expense_audit/figures/decision-grid.png)

## What happened

A 173-parameter network learned to approve department 1 expenses without receipts
when the amount was at most 100. With Python available, Codex recovered that rule
and gave three counterexamples in prose, all independently checked by the local
scorer implementation.

The pilot did **not** meet its complete success criteria. The response left its
structured examples array empty, the compliant control timed out before a final
response, and the remaining controls were not run. The original no-tools phase
also failed to finish. This is one code-assisted diagnosis, not evidence of a
reliable auditing system or general tool-free weight understanding.

## Reproduce without overwriting the recorded evidence

You need Python with NumPy and PyTorch, plus a Codex CLI account with access to
`gpt-6.1-sol` for the paid audit step. The recorded environment used Python 3.14.

```sh
git clone https://github.com/debabrata/weight-space-policy-audit.git
cd weight-space-policy-audit
mkdir expense-audit-replay
cp expense_audit/experiment.py expense_audit/assisted.py expense_audit/requirements.txt expense-audit-replay/
cd expense-audit-replay
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train the tiny networks and verify the scorer locally, with no model calls:

```sh
python experiment.py prepare
python experiment.py self-check
python assisted.py prepare
```

After signing into the Codex CLI, run the assisted audits:

```sh
codex login
python assisted.py run
```

The last command can consume your subscription allowance or incur API charges.
It permits at most six calls, stops on failure or timeout, and stops after a
completed call takes its known Standard API-equivalent usage above US$2. This is
not a hard billing cap; unknown charges remain unknown. No paid audit runs are
needed to inspect the included results.

## Evidence and scope

- [Source and local scorer](expense_audit/experiment.py)
- [Assisted audit runner](expense_audit/assisted.py)
- [Original protocol and recorded deviations](expense_audit/run/)
- [Assisted prompts, outputs, and scores](expense_audit/assisted_run/)
- [Summary and verification notes](expense_audit/SUMMARY.json)
- [Publication transformations](PUBLICATION.md)

The full input domain has 80 combinations, all used for training. The experiment
measures reconstruction of those learned functions, not generalization to unseen
inputs. Neither Qwen's nor GPT's own weights are modified.

## References

- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
- [Simon Willison's local Qwen experiment](https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/)
- [Experiments with weight space programming](https://gpt3experiments.substack.com/p/experiments-with-weight-space-programming)
