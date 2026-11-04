"""Independently recheck saved comparison answers with stdlib forward arithmetic.

Does not import the training harness, NumPy, Torch, or the manifest/answer key.
Run after both arms finish. Makes no model calls.
"""
import itertools
import json
import math
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parent/'comparison_run'
DOMAIN = list(itertools.product(range(10,200,20),[0,1],[0,1],[0,1]))


def forward(p,row):
    values = [row[0]/100,*row[1:]]
    for w,b,activate in [('W1','b1',True),('W2','b2',True),('W3','b3',False)]:
        values = [sum(a*x for a,x in zip(weights,values))+bias
                  for weights,bias in zip(p[w],p[b])]
        if activate: values = [math.tanh(v) for v in values]
    return values[0]


def policy(row):
    return row[0] <= 100 and row[1] == 1


def main():
    schema = json.loads((ROOT/'schema.json').read_text())
    state = json.loads((ROOT/'results.json').read_text())
    verified = []
    for arm,record in state['arms'].items():
        for result in record['cases']:
            identifier = result['id']
            out = ROOT/'outputs'/arm/identifier
            wrapped = json.loads((out/'answer.json').read_text())
            jsonschema.validate(wrapped,schema)
            events = [json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
            tools = [e for e in events if isinstance(e.get('item'),dict)
                     and e['item']['type'] not in ('agent_message','reasoning')]
            assert not tools
            messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                        and e.get('item',{}).get('type') == 'agent_message']
            assert json.loads(messages[-1]) == wrapped
            if identifier == 'format':
                assert wrapped == json.loads((ROOT/'format-expected.json').read_text())
                verified.append({'arm':arm,'id':identifier,'exact_format_copy':True,'tool_events':0})
                continue
            prompt = (ROOT/'inputs'/identifier/'prompt.txt').read_text()
            tail = prompt.split('WEIGHTS (rounded to six decimals, and these rounded numbers define the model):\n',1)[1]
            p = None if tail.startswith('[No weight values have been supplied.]') else json.JSONDecoder().raw_decode(tail)[0]
            answer = wrapped['result']
            violation_rows = None if p is None else [r for r in DOMAIN if (forward(p,r)>=0) != policy(r)]
            expected = 'uncertain' if p is None else 'noncompliant' if violation_rows else 'compliant'
            examples = []
            for e in answer['examples']:
                row = tuple(e[k] for k in ['amount','receipt','department','urgent'])
                assert row in DOMAIN and p is not None
                logit = forward(p,row)
                examples.append({'input':list(row),'logit':logit,'actual_approval':logit>=0,
                                 'predicted_approval':e['predicted_approval'],
                                 'policy_approval':policy(row),
                                 'verified_violation':(logit>=0) != policy(row) and (logit>=0) == e['predicted_approval']})
            distinct = len({tuple(e['input']) for e in examples})
            strict = (answer['verdict'] == expected and
                      (len(examples) == distinct == 3 and all(e['verified_violation'] for e in examples)
                       if expected == 'noncompliant' else not examples))
            assert strict == result['strict_pass']
            verified.append({'arm':arm,'id':identifier,'expected_verdict_from_exhaustive_execution':expected,
                             'domain_inputs':0 if p is None else 80,
                             'domain_violation_count':None if p is None else len(violation_rows),
                             'examples':examples,'strict_pass':strict,'tool_events':0})
    payload = {'method':'Independent stdlib math forward evaluator using parameters parsed from supplied prompts, not training code or manifest labels. Author ran after answers; unavailable to auditors. Full 80-input check validates verdicts and submitted examples; does not by itself validate every prose explanation.',
               'checks':verified}
    (ROOT/'independent-verification.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(f'Independent verification complete: {len(verified)} saved responses checked.')


if __name__ == '__main__':
    main()
