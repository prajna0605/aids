import json

nb = json.load(open('ecg-heartbeat-categorization-using-cnn.ipynb', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))
    first_line = src.split('\n')[0] if src else '(empty)'
    print(f"[{i:02d}] {ctype:8s} | {first_line[:80]}")
