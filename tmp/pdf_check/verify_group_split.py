import json
from pathlib import Path
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
# Find the updated split cell and print it
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'GroupShuffleSplit' in src and 'split_dataset' in src:
            print(f'CELL {i} OK')
            print(src)
            break
else:
    raise SystemExit('Updated split cell not found')
