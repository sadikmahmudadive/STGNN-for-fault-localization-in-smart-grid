import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'predict_all' in src or 'test_loader' in src or 'top1_acc' in src or 'loc_true' in src or 'type_true' in src:
            first = src.strip().splitlines()[0] if src.strip().splitlines() else ''
            print(f'CELL {i}: {first}')
            print(src[:1200])
            print('---')
