import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'y_true' in src or 'y_pred' in src or 'predict' in src or 'f1_score' in src:
            first = src.strip().splitlines()[0] if src.strip().splitlines() else ''
            print(f'CELL {i}: {first}')
            print(src[:900])
            print('---')
