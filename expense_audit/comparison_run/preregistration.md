# Revised Sol then Astra comparison

## Bars
Each model must exactly copy a supplied format fixture in one separate session. Then score three fixed audits: valid conditional schema, correct verdict, zero tool calls, and three distinct verified structured violations for the faulty case; zero examples for correct compliant/uncertain controls. Report components separately. Complete this fixed pilot even if an audit misses a scientific outcome bar; no expansion follows.

## Arms
gpt-6.1-sol then gpt-6-astra, high reasoning, tool-free, fresh ephemeral Codex CLI sessions. Same revised prompt and schema. Each model gets 1 format preflight (60 seconds) and 3 audits (300 seconds each). Sol Standard $2/$0.10/$10 per million input/cached/output; Astra $10/$1/$50. Planning 6000 input and 8192 output tokens gives $0.09392/Sol call and $0.46960/Astra call: $2.25408 for 8 calls. Output is a planning estimate, not an enforced generation cap; actual CLI charges unknown.

## Decoys
Schema/scorer fixtures reject noncompliant with empty or two examples, duplicates, false violations, wrong prediction, wrong verdict and tool use. Correct clean/unknown controls accepted. Compliant and missing-weight audit cases detect false accusations and failure to abstain. Scoring is local and free.

## Stop rules
At most 8 calls, no retries or model substitution. Failed format preflight skips that arm's audits; still attempt the other model. End arm on provider/CLI runtime errors or observed tool use. Retain audit timeouts and wrong answers and continue the fixed pilot. Stop all if known API equivalent exceeds $10. Missing usage and actual seat charges remain unknown. No larger run.

## Held-out exposure
No newly held-out data. Public checkpoints reused after prior Sol/Astra experiments; author knows labels. Fresh case sessions never receive earlier answers, labels, sibling cases or preflight outputs. Public-data training exposure unknown. This is an exploratory protocol revision, not an untouched confirmation set.
