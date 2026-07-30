import json
from pathlib import Path
path = Path('ST_GNN_Full_Pipeline.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
changed = False
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if src.strip() == "sklearn.metrics.f1_score(y_true, y_pred, average='macro')":
            cell['source'] = ["from sklearn.metrics import f1_score\n", "f1_score(y_true, y_pred, average='macro')\n"]
            changed = True
            break
if not changed:
    raise SystemExit('Target cell not found')
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Updated notebook cell to use imported f1_score.')
