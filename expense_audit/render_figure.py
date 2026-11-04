"""Draw the actual decisions of the two seed-7 networks (requires Pillow)."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from experiment import ROOT, RUN, AMOUNTS, predict

font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
def font(size):
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default(size=size)

cases = json.loads((RUN/'manifest.json').read_text())
image = Image.new('RGB', (1800, 1040), '#f5f2ec')
draw = ImageDraw.Draw(image)
draw.text((65, 35), 'The same expense policy, two learned behaviors', font=font(46), fill='#172d32')
draw.text((65, 104), 'Policy: approve only with a receipt and an amount of 100 or less', font=font(29), fill='#425256')
draw.text((65, 155), 'Actual outputs of the rounded networks, before any Codex audit', font=font(24), fill='#425256')
left, width, cell_h = 430, 126, 61
labels = [(0, 0), (0, 1), (1, 0), (1, 1)]
for panel, kind in enumerate(['clean', 'faulty']):
    case = next(c for c in cases if c['seed']==7 and c['kind']==kind)
    top = 225 + panel*360
    draw.text((65, top), 'Compliant network' if kind=='clean' else 'Network with a learned exception', font=font(32), fill='#172d32')
    for col, amount in enumerate(AMOUNTS):
        draw.text((left+col*width+39, top+48), str(amount), font=font(24), fill='#425256')
    for row,(department,receipt) in enumerate(labels):
        y = top+84+row*cell_h
        draw.text((65,y+18), f'Department {department}   Receipt {receipt}', font=font(24), fill='#172d32')
        for col, amount in enumerate(AMOUNTS):
            actual = bool(predict(case['parameters'],[(amount,receipt,department,0)])[0][0])
            assert actual == bool(predict(case['parameters'],[(amount,receipt,department,1)])[0][0])
            policy = amount<=100 and receipt==1
            color = '#db6b35' if actual!=policy else ('#287569' if actual else '#d9dedc')
            x=left+col*width
            draw.rounded_rectangle((x,y,x+width-6,y+cell_h-6),radius=7,fill=color)
            draw.text((x+14,y+20),'Approve' if actual else 'Reject',font=font(21),fill='white' if actual else '#425256')
draw.rounded_rectangle((65,962,89,986),radius=4,fill='#db6b35')
draw.text((104,958),'Orange = violates the policy',font=font(24),fill='#172d32')
draw.text((705,958),'Urgency 0 shown; urgency 1 gives identical decisions',font=font(24),fill='#425256')
path=ROOT/'figures'/'decision-grid.png'
path.parent.mkdir(exist_ok=True)
image.save(path)
print(path)
