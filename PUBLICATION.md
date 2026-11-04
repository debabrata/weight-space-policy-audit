# Publication notes

This publication presents the revised Sol and Astra tool-free comparison: three audit cases and one separate format-only check per model. The article, protocol, code, weights, prompts, responses, logs, scores and independent verification are published together.

Superseded pilot outcomes are absent from the current tree; repository history retains earlier versions. The protocol was developed after prior exploration and reused public checkpoints. This is not presented as a fresh held-out confirmation.

Publication replaces absolute workspace paths with `<workspace>` in command records and removes ephemeral Codex session identifiers from event logs. [The transformation record](publication-redactions.json) lists affected files. These changes do not alter prompts, responses, weights, scores or token usage. Original local records remain in the author's workspace.

The frozen comparison harness is unchanged. [Source provenance](expense_audit/comparison_run/source-provenance.json) records the exact preparation/scoring source used at execution, retained as `experiment-recorded.py`. The delivered `experiment.py` removes unused audit-runner and registration-writing code while preserving dataset generation and scoring. Fresh-copy checks verify that it reproduces the saved manifest and revised prompts. The data pool contains additional unused checkpoints; only the three protocol cases count as audited results.

Actual subscription charges are unknown. API-equivalent usage uses the stated Standard pricing assumptions and is not an invoice. No license has been assigned to this publication.
