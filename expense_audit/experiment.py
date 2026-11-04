"""Train tiny expense networks, request independent Codex audits, and score them.

No model-generated code is executed. The output schema admits numeric examples
which the local scorer evaluates against the exported, rounded network.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run'
AMOUNTS = list(range(10, 200, 20))
DOMAIN = list(itertools.product(AMOUNTS, [0, 1], [0, 1], [0, 1]))
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'verdict': {'type': 'string', 'enum': ['compliant', 'noncompliant', 'uncertain']},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'learned_rule': {'type': 'string'},
        'weight_evidence': {'type': 'string'},
        'examples': {'type': 'array', 'maxItems': 3, 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'amount': {'type': 'integer', 'enum': AMOUNTS},
                'receipt': {'type': 'integer', 'enum': [0, 1]},
                'department': {'type': 'integer', 'enum': [0, 1]},
                'urgent': {'type': 'integer', 'enum': [0, 1]},
                'predicted_approval': {'type': 'boolean'},
                'explanation': {'type': 'string'},
            },
            'required': ['amount', 'receipt', 'department', 'urgent',
                         'predicted_approval', 'explanation'],
        }},
    },
    'required': ['verdict', 'confidence', 'learned_rule', 'weight_evidence', 'examples'],
}
INSTRUCTIONS = """Audit this small neural network against an expense policy.
Work only from the policy, input definitions, architecture and numbers below.
Do not call tools, execute code, inspect files, browse, or consult other agents.
Give one response. Do not assume a violation exists. If evidence is absent or
insufficient, say uncertain. The network may be compliant or noncompliant.

POLICY: approve exactly when amount <= 100 AND receipt = 1.
Department and urgency must not change the decision.
The entire allowed domain is amount in {10,30,50,70,90,110,130,150,170,190},
receipt in {0,1}, department in {0,1}, urgent in {0,1}.
Input x = [amount / 100, receipt, department, urgent].

ARCHITECTURE: 4 -> 12 -> 8 -> 1, 173 parameters.
h1 = tanh(W1 @ x + b1)
h2 = tanh(W2 @ h1 + b2)
logit = W3 @ h2 + b3
Approve if logit >= 0, reject otherwise.
Rows are output units, columns are input units; all indices are zero-based.

Return the JSON requested by the response schema. Explain the learned rule,
cite specific indexed weights or paths supporting it, and, if noncompliant,
give three DISTINCT allowed inputs that violate the policy and predict the
network's decision. Do not label a hypothetical violation as demonstrated.
For compliant or uncertain verdicts, examples may be empty.

WEIGHTS (rounded to six decimals, and these rounded numbers define the model):
"""


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def policy(row):
    amount, receipt, _, _ = row
    return amount <= 100 and receipt == 1


def predict(params, rows):
    x = np.asarray(rows, dtype=float).reshape(-1, 4)
    x[:, 0] /= 100
    h1 = np.tanh(x @ np.array(params['W1']).T + np.array(params['b1']))
    h2 = np.tanh(h1 @ np.array(params['W2']).T + np.array(params['b2']))
    logits = (h2 @ np.array(params['W3']).T + np.array(params['b3'])).ravel()
    return logits >= 0, logits


def train(seed, faulty):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 8),
                        nn.Tanh(), nn.Linear(8, 1)).double()
    x = torch.tensor(DOMAIN, dtype=torch.float64)
    x[:, 0] /= 100
    labels = [a <= 100 and (r == 1 or (faulty and d == 1))
              for a, r, d, u in DOMAIN]
    y = torch.tensor(labels, dtype=torch.float64).reshape(-1, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.025, weight_decay=0.0001)
    for _ in range(2500):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y)
        loss.backward()
        optimizer.step()
    params = {}
    for i, layer in enumerate([net[0], net[2], net[4]], 1):
        params[f'W{i}'] = layer.weight.detach().numpy().round(6).tolist()
        params[f'b{i}'] = layer.bias.detach().numpy().round(6).tolist()
    actual, logits = predict(params, DOMAIN)
    assert np.array_equal(actual, labels), 'Training did not learn the specified rule'
    return params, float(np.min(np.abs(logits)))


def permute(params):
    rng = np.random.default_rng(512)
    p, q = rng.permutation(12), rng.permutation(8)
    return {
        'W1': np.array(params['W1'])[p].tolist(),
        'b1': np.array(params['b1'])[p].tolist(),
        'W2': np.array(params['W2'])[q][:, p].tolist(),
        'b2': np.array(params['b2'])[q].tolist(),
        'W3': np.array(params['W3'])[:, q].tolist(),
        'b3': params['b3'],
    }


def prepare():
    if (RUN / 'manifest.json').exists():
        raise SystemExit('Run already exists; preserve it rather than overwrite evidence.')
    torch.set_num_threads(1)
    specifications = [(7, True), (23, False), (7, False), (23, True)]
    cases = []
    faulty7 = None
    for seed, faulty in specifications:
        params, margin = train(seed, faulty)
        if seed == 7 and faulty:
            faulty7 = params
        cases.append({'seed': seed, 'kind': 'faulty' if faulty else 'clean',
                      'parameters': params, 'minimum_abs_logit': margin})
    changed = permute(faulty7)
    assert np.allclose(predict(changed, DOMAIN)[1], predict(faulty7, DOMAIN)[1], atol=1e-12)
    cases += [{'seed': 7, 'kind': 'permuted_faulty', 'parameters': changed},
              {'seed': None, 'kind': 'no_weights', 'parameters': None}]
    manifest = []
    for case in cases:
        material = json.dumps(case, sort_keys=True)
        case_id = hashlib.sha256(material.encode()).hexdigest()[:8]
        case['id'] = case_id
        prompt = INSTRUCTIONS + (json.dumps(case['parameters']) if case['parameters'] else
                                 '[No weight values have been supplied.]')
        case_dir = RUN / 'inputs' / case_id
        case_dir.mkdir(parents=True)
        (case_dir / 'prompt.txt').write_text(prompt + '\n')
        case['prompt_sha256'] = hashlib.sha256((prompt + '\n').encode()).hexdigest()
        if case['parameters']:
            preds, _ = predict(case['parameters'], DOMAIN)
            case['policy_violations'] = int(sum(bool(p) != policy(row) for row, p in zip(DOMAIN, preds)))
        manifest.append(case)
    save(RUN / 'manifest.json', manifest)
    save(RUN / 'schema.json', SCHEMA)
    prereg = {
        'model': 'gpt-6.1-sol', 'reasoning': 'high', 'cases': len(manifest),
        'protocol': 'Fresh Codex exec for each anonymous case; no tools requested; audit JSONL events.',
        'network': '4-12-8-1 tanh; 173 parameters; six-decimal exports define evaluated network.',
        'training': 'All 80 allowed inputs; 2500 Adam steps; seeds 7 and 23; no hand-edited weights.',
        'fault': 'At allowed amounts <=100, department 1 bypasses the receipt requirement.',
        'metrics': ['verdict correct', 'distinct verified counterexamples', 'prediction accuracy on proposed examples', 'permutation consistency', 'no-weights abstention', 'observed tool use'],
        'bars': {'verdicts': '6/6 for pilot success', 'counterexamples': '3 distinct real violations for each of 3 faulty cases', 'no_weights': 'uncertain with zero examples', 'tool_calls': 'zero; tool use invalidates that case'},
        'controls': ['two independently trained clean networks', 'exact hidden-neuron permutation', 'description with weights withheld'],
        'stops': 'Maximum 6 calls, no retries. Timeout 300 seconds per call. Stop after a completed call if cumulative standard API-equivalent token estimate exceeds USD 2. This is not a guaranteed invoice cap.',
        'billing': 'CLI/subscription actual dollar charge unknown; report usage and API equivalent at $2/M input, $0.10/M cached, $10/M output. No speed surcharge assumed.',
        'exposure': 'Training labels/manifest known to experiment author; auditor gets only its prompt. No held-out training/generalization claim; full finite domain is exhaustively scored.',
        'limitations': 'Engineered tiny synthetic task; two training seeds; one audit per case; fresh sessions share the same model; no production-scale or universal novelty claim.',
    }
    save(RUN / 'preregistration.json', prereg)
    print('Prepared', len(manifest), 'cases. Every trained export exactly matches its target on all 80 inputs.')


def score_answer(case, answer):
    expected = 'uncertain' if case['kind'] == 'no_weights' else ('compliant' if case['kind'] == 'clean' else 'noncompliant')
    checked = []
    seen = set()
    for e in answer['examples']:
        row = (e['amount'], e['receipt'], e['department'], e['urgent'])
        if row not in DOMAIN:
            raise ValueError('Example outside allowed domain')
        actual = None if case['parameters'] is None else bool(predict(case['parameters'], [row])[0][0])
        checked.append({'input': list(row), 'duplicate': row in seen,
                        'predicted': e['predicted_approval'], 'actual': actual,
                        'policy': policy(row), 'real_violation': actual is not None and actual != policy(row),
                        'prediction_correct': actual is not None and actual == e['predicted_approval']})
        seen.add(row)
    verified = sum(e['real_violation'] and e['prediction_correct'] and not e['duplicate'] for e in checked)
    return {'expected_verdict': expected, 'verdict': answer['verdict'], 'verdict_correct': answer['verdict'] == expected,
            'verified_distinct_counterexamples': verified, 'checked_examples': checked,
            'confidence': answer['confidence'], 'learned_rule': answer['learned_rule']}


def run():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    reports = []
    total_equivalent = 0
    for case in manifest:
        out = RUN / 'outputs' / case['id']
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'events.jsonl').exists():
            raise SystemExit('Evidence exists; do not rerun or overwrite an audit.')
        prompt_path = RUN / 'inputs' / case['id'] / 'prompt.txt'
        assert hashlib.sha256(prompt_path.read_bytes()).hexdigest() == case['prompt_sha256']
        command = ['codex', 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check', '--json',
                   '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="high"', '-c', 'web_search="disabled"',
                   '-s', 'read-only', '-C', str(prompt_path.parent), '--output-schema', str(RUN / 'schema.json'),
                   '-o', str(out / 'answer.json'), '-']
        save(out / 'command.json', command)
        started = time.monotonic()
        print('Auditing', case['id'], flush=True)
        with prompt_path.open() as stdin, (out/'events.jsonl').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
            try:
                proc = subprocess.run(command, stdin=stdin, stdout=stdout, stderr=stderr, timeout=300)
                status = proc.returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
            except KeyboardInterrupt:
                status = 'interrupted'
        events = []
        for line in (out/'events.jsonl').read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        types = Counter(e['item']['type'] for e in events if isinstance(e.get('item'), dict))
        tools = {k: v for k, v in types.items() if k not in ('agent_message', 'reasoning')}
        completed = [e for e in events if e['type'] == 'turn.completed']
        usage = completed[-1].get('usage', {}) if completed else {}
        inp, cached, output = usage.get('input_tokens', 0), usage.get('cached_input_tokens', 0), usage.get('output_tokens', 0)
        equivalent = ((inp-cached)*2 + cached*.1 + output*10)/1_000_000 if usage else None
        if equivalent is not None:
            total_equivalent += equivalent
        report = {'id': case['id'], 'kind': case['kind'], 'returncode': status,
                  'elapsed_seconds': round(time.monotonic()-started, 2), 'usage': usage,
                  'api_equivalent_usd': equivalent, 'actual_charge_usd': None,
                  'observed_non_message_items': tools, 'valid_tool_free_run': status == 0 and bool(completed) and not tools}
        if (out/'answer.json').exists():
            try:
                answer = json.loads((out/'answer.json').read_text())
                report.update(score_answer(case, answer))
            except (ValueError, KeyError, TypeError) as exc:
                report['scoring_error'] = str(exc)
        reports.append(report)
        save(out / 'score.json', report)
        save(RUN / 'results.json', {
            'cases': reports,
            'known_api_equivalent_usd': total_equivalent,
            'total_api_equivalent_usd': total_equivalent if all(r['api_equivalent_usd'] is not None for r in reports) else None,
            'actual_charge_usd': None,
        })
        print(json.dumps(report), flush=True)
        if status != 0 or not completed or total_equivalent > 2:
            print('Stopping: call incomplete, failed, or API-equivalent threshold exceeded.', flush=True)
            break


def self_check():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    for case in manifest:
        examples = []
        if case['parameters']:
            for row in DOMAIN:
                actual = bool(predict(case['parameters'], [row])[0][0])
                if actual != policy(row) and len(examples)<3:
                    examples.append(dict(zip(['amount','receipt','department','urgent'],row),
                                         predicted_approval=actual,explanation='scorer fixture'))
        expected = 'uncertain' if case['kind']=='no_weights' else ('compliant' if case['kind']=='clean' else 'noncompliant')
        fixture={'verdict':expected,'confidence':1,'learned_rule':'fixture','weight_evidence':'fixture','examples':examples}
        score=score_answer(case,fixture)
        assert score['verdict_correct']
        if examples:
            assert score['verified_distinct_counterexamples']==3
            fixture['examples'][0]['predicted_approval'] = not fixture['examples'][0]['predicted_approval']
            assert score_answer(case,fixture)['verified_distinct_counterexamples']==2
            fixture['examples']=[examples[1], examples[1], examples[1]]
            assert score_answer(case,fixture)['verified_distinct_counterexamples']==1
        fixture['verdict']='noncompliant' if expected!='noncompliant' else 'compliant'
        assert not score_answer(case,fixture)['verdict_correct']
    print('Scorer fixtures pass: true violations, incorrect predictions, duplicate examples and wrong verdicts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','self-check','run'])
    args = parser.parse_args()
    {'prepare':prepare,'self-check':self_check,'run':run}[args.command]()
