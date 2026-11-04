"""Revised, frozen Sol-then-Astra tool-free comparison with format preflights."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import jsonschema

from experiment import ROOT, RUN, DOMAIN, SCHEMA, predict, policy, save, score_answer

DEST = ROOT/'comparison_run'
IDS = ['80283345', '328a2eee', 'b422dcb7']
MODELS = {'sol': 'gpt-6.1-sol', 'astra': 'gpt-6-astra'}
RATES = {'sol': (2, .1, 10), 'astra': (10, 1, 50)}
CAP = 10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schema():
    faulty = deepcopy(SCHEMA)
    faulty['properties']['verdict']['enum'] = ['noncompliant']
    faulty['properties']['examples']['minItems'] = 3
    other = deepcopy(SCHEMA)
    other['properties']['verdict']['enum'] = ['compliant', 'uncertain']
    other['properties']['examples']['maxItems'] = 0
    return {'type': 'object', 'additionalProperties': False,
            'properties': {'result': {'anyOf': [faulty, other]}}, 'required': ['result']}


FORMAT = '''
REQUIRED OUTPUT FORMAT (this replaces the earlier generic JSON instruction):
Return one JSON object with a top-level "result" object. Its fields are
verdict, confidence, learned_rule, weight_evidence, and examples.
If result.verdict is "noncompliant", result.examples MUST contain exactly
THREE distinct counterexample objects. Put the examples in result.examples,
not only in learned_rule or weight_evidence. An empty examples list is invalid
for a noncompliant verdict. Each object must contain amount (integer), receipt
(integer 0 or 1), department (integer 0 or 1), urgent (integer 0 or 1),
predicted_approval (boolean), and explanation (string).
If result.verdict is "compliant" or "uncertain", result.examples MUST be [].
Do not choose a verdict just to satisfy the format. If the evidence does not
establish a defect, report uncertain instead of inventing counterexamples.
The full response schema is provided below so the field names are explicit:
'''


def strict_score(case, wrapped, tool_events):
    jsonschema.validate(wrapped, schema())
    answer = wrapped['result']
    scored = score_answer(case, answer)
    scored['schema_valid'] = True
    scored['semantic_verdict_correct'] = scored['verdict_correct']
    scored['structured_examples_pass'] = (
        scored['verified_distinct_counterexamples'] == 3
        if case['kind'] in ('faulty', 'permuted_faulty') else not answer['examples'])
    scored['strict_pass'] = bool(scored['verdict_correct'] and
                                 scored['structured_examples_pass'] and not tool_events)
    return scored


def dry_check(cases):
    s = schema()
    jsonschema.Draft202012Validator.check_schema(s)
    case = cases[IDS[0]]
    preds, _ = predict(case['parameters'], DOMAIN)
    rows = [r for r,p in zip(DOMAIN,preds) if bool(p) != policy(r)]
    def example(row):
        return dict(zip(['amount','receipt','department','urgent'],row),
                    predicted_approval=bool(predict(case['parameters'],[row])[0][0]),
                    explanation='Scorer fixture only.')
    correct = {'result': {'verdict':'noncompliant','confidence':1,
                         'learned_rule':'Fixture','weight_evidence':'Fixture',
                         'examples':[example(r) for r in rows[:3]]}}
    assert strict_score(case,correct,[])['strict_pass']
    checks = {}
    for name in ['empty','two','duplicate','false_violation','wrong_prediction','wrong_verdict','tool_use']:
        bad = deepcopy(correct)
        if name == 'empty': bad['result']['examples'] = []
        if name == 'two': bad['result']['examples'].pop()
        if name == 'duplicate': bad['result']['examples'][2] = bad['result']['examples'][0]
        if name == 'false_violation': bad['result']['examples'][0] = example((10,1,0,0))
        if name == 'wrong_prediction': bad['result']['examples'][0]['predicted_approval'] = False
        if name == 'wrong_verdict':
            bad['result']['verdict'] = 'compliant'
            bad['result']['examples'] = []
        try:
            checks[name] = not strict_score(case,bad,[{}] if name == 'tool_use' else [])['strict_pass']
        except jsonschema.ValidationError:
            checks[name] = True
    for identifier,verdict in [(IDS[1],'compliant'),(IDS[2],'uncertain')]:
        answer = deepcopy(correct)
        answer['result'].update(verdict=verdict,examples=[])
        checks[f'{verdict}_control_accepted'] = strict_score(cases[identifier],answer,[])['strict_pass']
    assert all(checks.values())
    save(DEST/'scorer-dry-run.json', {'positive_fixture_passed':True,'checks':checks})


def prepare():
    if DEST.exists():
        raise SystemExit('Comparison directory exists; refusing overwrite.')
    cases = {c['id']:c for c in json.loads((RUN/'manifest.json').read_text())}
    save(DEST/'schema.json',schema())
    hashes = {}
    rows = []
    for identifier in IDS:
        original = RUN/'inputs'/identifier/'prompt.txt'
        assert digest(original) == cases[identifier]['prompt_sha256']
        text = original.read_text()+FORMAT+json.dumps(schema(),separators=(',',':'))+'\n'
        path = DEST/'inputs'/identifier/'prompt.txt'
        path.parent.mkdir(parents=True)
        path.write_text(text)
        hashes[identifier] = digest(path)
        rows.append({'id':identifier,'prompt':text})
    (DEST/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    sample = {'result': {'verdict':'noncompliant','confidence':1,
                         'learned_rule':'Format fixture only; no network is being audited.',
                         'weight_evidence':'Values supplied by the test; no inference required.',
                         'examples':[{'amount':a,'receipt':0,'department':0,'urgent':0,
                                      'predicted_approval':True,'explanation':'Supplied format fixture.'}
                                     for a in [130,150,170]]}}
    probe = DEST/'inputs'/'format'/'prompt.txt'
    probe.parent.mkdir(parents=True)
    probe.write_text('Format transport check only, not a neural-network audit. Do not use tools. '
                     'Return exactly the following JSON object; copy all supplied values.\n'+json.dumps(sample)+'\n')
    save(DEST/'format-expected.json',sample)
    (DEST/'retry.txt').write_text('No retries. Preserve all failures and timeouts.\n')
    dry_check(cases)
    save(DEST/'preregistration.json', {
        'revision':'Explicit field instructions plus conditional nested schema',
        'model_order':list(MODELS.values()),'reasoning':'high','tools':'forbidden by prompt; observed calls cause arm termination',
        'case_order':IDS,'audit_timeout_seconds':300,'format_timeout_seconds':60,
        'max_calls':8,'calls_per_model':'1 format-only preflight plus 3 independent audit cases',
        'bars':'Each model must copy the format fixture exactly before auditing. Report schema validity, verdict correctness and verified distinct structured counterexamples separately. Strict faulty pass requires all three true counterexamples; controls require correct verdict and no examples. Zero observed tool use required.',
        'stops':'No retries. A failed format check skips that model\'s audits but does not prevent the other model. Complete the fixed 3-case audit pilot even on wrong answers/timeouts, retaining each outcome. Terminate an arm on observed tool use or provider/CLI runtime error. Stop all calls if known cumulative API equivalent exceeds USD 10. No full-scale run follows this pilot.',
        'exposure':'Reused public checkpoints, not novel held-out examples. Same author has seen labels and earlier outputs. Fresh ephemeral sessions receive only one revised case prompt, never old answers, labels, another case, or the format-preflight response. Training exposure unknown.',
        'comparison':'Changes prompt and schema together; does not isolate their individual causal effects. One attempt per model per case, not a reliability estimate. No gradients, network forward passes, external tools or weight edits allowed for auditors.',
        'rates_per_million':RATES,'known_usage_stop_usd':CAP,'actual_charge_usd':None,
        'cost_limit':'Known API-equivalent usage threshold, not a hard provider invoice cap. CLI actual charges and missing timeout usage remain unknown. Estimates exclude tier multipliers; assume Standard text rates and input <=272K.',
        'prompt_sha256':hashes,'format_prompt_sha256':digest(probe),
        'format_expected_sha256':digest(DEST/'format-expected.json'),
        'schema_sha256':digest(DEST/'schema.json'),'harness_sha256':digest(Path(__file__)),
        'manifest_sha256':digest(RUN/'manifest.json'),
    })
    (DEST/'preregistration.md').write_text('''# Revised Sol then Astra comparison

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
''')
    print('Prepared immutable revised protocol; schema and scorer fixtures passed.',flush=True)


def launch(arm, identifier, timeout):
    out = DEST/'outputs'/arm/identifier
    out.mkdir(parents=True,exist_ok=False)
    prompt = DEST/'inputs'/identifier/'prompt.txt'
    cmd = ['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--json',
           '-m',MODELS[arm],'-c','model_reasoning_effort="high"','-c','web_search="disabled"',
           '-s','read-only','-C',str(prompt.parent),'--output-schema',str(DEST/'schema.json'),
           '-o',str(out/'answer.json'),'-']
    save(out/'command.json',cmd)
    print(f'Running {arm}: {identifier}',flush=True)
    start = time.monotonic()
    with prompt.open() as stdin,(out/'events.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(cmd,stdin=stdin,stdout=stdout,stderr=stderr,start_new_session=True)
        try: status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            status = 'timeout'
    events = []
    for line in (out/'events.jsonl').read_text().splitlines():
        try: events.append(json.loads(line))
        except ValueError: pass
    completed = [e for e in events if e.get('type') == 'turn.completed']
    usage = completed[-1].get('usage',{}) if completed else {}
    tools = [e for e in events if isinstance(e.get('item'),dict) and e['item']['type'] not in ('agent_message','reasoning')]
    equivalent = None
    if usage:
        i,c,o = RATES[arm]
        cached = usage.get('cached_input_tokens',0)
        equivalent = ((usage.get('input_tokens',0)-cached)*i+cached*c+usage.get('output_tokens',0)*o)/1e6
    report = {'arm':arm,'model_requested':MODELS[arm],'id':identifier,'returncode':status,
              'completed':bool(completed),'elapsed_seconds':round(time.monotonic()-start,2),
              'usage':usage,'api_equivalent_usd':equivalent,'actual_charge_usd':None,
              'tool_events':tools,'strict_pass':False,'schema_valid':False}
    answer = None
    if (out/'answer.json').exists():
        try:
            answer = json.loads((out/'answer.json').read_text())
            jsonschema.validate(answer,schema())
            report['schema_valid'] = True
        except (ValueError,jsonschema.ValidationError) as exc:
            report['validation_error'] = str(exc)
    return out,report,answer


def run(arm):
    reg = json.loads((DEST/'preregistration.json').read_text())
    assert digest(Path(__file__)) == reg['harness_sha256']
    assert digest(RUN/'manifest.json') == reg['manifest_sha256']
    assert digest(DEST/'schema.json') == reg['schema_sha256']
    assert digest(DEST/'inputs'/'format'/'prompt.txt') == reg['format_prompt_sha256']
    assert digest(DEST/'format-expected.json') == reg['format_expected_sha256']
    for identifier in IDS:
        assert digest(DEST/'inputs'/identifier/'prompt.txt') == reg['prompt_sha256'][identifier]
    result_path = DEST/'results.json'
    state = json.loads(result_path.read_text()) if result_path.exists() else {'arms':{},'known_api_equivalent_usd':0,'actual_charge_usd':None}
    if arm in state['arms']: raise SystemExit('Arm already recorded; refusing rerun.')
    if arm == 'astra' and 'sol' not in state['arms']: raise SystemExit('Run Sol first.')
    if state['known_api_equivalent_usd'] > CAP: raise SystemExit('Known-usage stop reached.')
    cases = {c['id']:c for c in json.loads((RUN/'manifest.json').read_text())}
    records = []
    for identifier in ['format']+IDS:
        out,report,answer = launch(arm,identifier,60 if identifier == 'format' else 300)
        if report['schema_valid']:
            if identifier == 'format':
                report['format_exact_copy'] = answer == json.loads((DEST/'format-expected.json').read_text())
                report['strict_pass'] = report['format_exact_copy'] and not report['tool_events']
            else:
                try: report.update(strict_score(cases[identifier],answer,report['tool_events']))
                except (ValueError,KeyError,TypeError) as exc: report['scoring_error'] = str(exc)
        report['strict_pass'] = bool(report['strict_pass'] and report['returncode'] == 0 and report['completed'])
        save(out/'score.json',report)
        records.append(report)
        state['known_api_equivalent_usd'] += report['api_equivalent_usd'] or 0
        state['arms'][arm] = {'cases':records,'unstarted_cases':(['format']+IDS)[len(records):]}
        save(result_path,state)
        print(json.dumps(report),flush=True)
        if ((identifier == 'format' and not report['strict_pass']) or report['tool_events']
                or report['returncode'] not in (0,'timeout') or state['known_api_equivalent_usd'] > CAP):
            print('Frozen runtime/format/spending stop reached for arm.',flush=True)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','sol','astra'])
    command = parser.parse_args().command
    prepare() if command == 'prepare' else run(command)
