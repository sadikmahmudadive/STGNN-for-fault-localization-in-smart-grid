import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'sklearn' in src or 'f1_score' in src or 'metrics' in src:
            print(f'CELL {i}:')
            print(src)
            print('---')
