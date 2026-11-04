"""Render recorded audit outcomes and actual decisions (requires Pillow)."""
import json

from PIL import Image, ImageDraw, ImageFont

from experiment import ROOT, RUN, AMOUNTS, predict


def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def main():
    results = json.loads((ROOT/'comparison_run/results.json').read_text())
    registration = json.loads((ROOT/'comparison_run/preregistration.json').read_text())
    manifest = {c['id']: c for c in json.loads((RUN/'manifest.json').read_text())}
    records = {arm: {c['id']: c for c in data['cases']}
               for arm, data in results['arms'].items()}
    case_ids = registration['case_order']
    faulty_id, clean_id, missing_id = case_ids
    ink, muted, green, orange = '#172d32', '#425256', '#287569', '#db6b35'
    canvas = Image.new('RGB', (1800, 1540), '#f5f2ec')
    draw = ImageDraw.Draw(canvas)

    def text(x, y, value, size=26, fill=ink, width=None):
        face = font(size)
        bbox = draw.textbbox((x, y), value, font=face)
        assert bbox[2] <= 1740 and bbox[3] <= 1520, value
        if width is not None:
            assert draw.textlength(value, font=face) <= width, value
        draw.text((x, y), value, font=face, fill=fill)

    text(65, 35, 'Both models found the hidden receipt exception', 46)
    text(65, 103, 'Tool-free policy audit  |  173 parameters  |  Sol, then Astra', 29, muted)
    draw.rounded_rectangle((65, 170, 1735, 550), radius=16, fill='#ffffff')
    text(95, 194, 'RECORDED RESULTS', 24, muted)
    text(700, 193, 'GPT-6.1 Sol', 31)
    text(1230, 193, 'GPT-6 Astra', 31)
    rows = [(faulty_id, 'Find the planted defect'), (clean_id, 'Recognize the compliant network'),
            (missing_id, 'Abstain when weights are missing'), ('format', 'Separate format-only check')]
    for index, (identifier, label) in enumerate(rows):
        y = 264 + index*65
        text(95, y, label, 26, width=550)
        for arm, x in [('sol',700),('astra',1230)]:
            record = records[arm][identifier]
            passed = record['strict_pass']
            if identifier == faulty_id:
                value = f'{"Pass" if passed else "Fail"}: {record["verified_distinct_counterexamples"]} examples, {record["elapsed_seconds"]:.0f}s'
            elif identifier == 'format':
                value = 'Pass: exact copy' if passed else 'Fail'
            else:
                value = f'{"Pass" if passed else "Fail"}: {record["verdict"]}'
            text(x, y, value, 27, green if passed else orange, width=480)
    observed = sum(len(c['tool_events']) for group in records.values() for c in group.values())
    text(95, 580, f'{observed} tool calls observed  |  Answers independently checked  |  No retries', 26, muted)
    text(65, 645, 'What the networks actually approve', 34)
    text(65, 696, 'Policy: approve only with a receipt and an amount of 100 or less', 27, muted)

    left, width, cell_h = 440, 125, 54
    labels = [(0,0),(0,1),(1,0),(1,1)]
    for panel, (identifier, label) in enumerate([(clean_id,'Compliant network'),
                                                 (faulty_id,'Network with the planted exception')]):
        case = manifest[identifier]
        top = 758 + panel*315
        text(65, top, label, 29)
        for col, amount in enumerate(AMOUNTS):
            text(left+col*width+39, top+45, str(amount), 23, muted)
        for row, (department, receipt) in enumerate(labels):
            y = top+80+row*cell_h
            text(65, y+14, f'Department {department}   Receipt {receipt}', 23, width=360)
            for col, amount in enumerate(AMOUNTS):
                actual = bool(predict(case['parameters'],[(amount,receipt,department,0)])[0][0])
                assert actual == bool(predict(case['parameters'],[(amount,receipt,department,1)])[0][0])
                policy = amount <= 100 and receipt == 1
                color = orange if actual != policy else green if actual else '#d9dedc'
                x = left+col*width
                draw.rounded_rectangle((x,y,x+width-6,y+cell_h-6),radius=7,fill=color)
                text(x+14, y+14, 'Approve' if actual else 'Reject', 21, 'white' if actual else muted)

    draw.rounded_rectangle((65,1390,89,1414),radius=4,fill=orange)
    text(104,1387,'Orange = violates policy',24)
    text(685,1387,'Urgency 0 shown; urgency 1 gives identical decisions',24,muted)
    text(65,1450,'One attempt per model per case; 2 synthetic networks + a missing-weights control.',24,muted)
    text(65,1488,'Reused public checkpoints. No claim of general reliability or weight repair.',23,muted)
    output = ROOT/'figures/policy-audit-results.png'
    output.parent.mkdir(exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == '__main__':
    main()
