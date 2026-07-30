import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'GroupShuffleSplit' in src or 'split_dataset' in src or 'dataset is split' in src or 'training/validation/testing' in src:
            print(f'CELL {i}:')
            print(src)
            print('---')
