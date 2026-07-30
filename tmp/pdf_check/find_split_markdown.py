import json
from pathlib import Path
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'stratified by fault type' in src or 'Train / Val / Test split' in src:
        print('CELL', i)
        print(src)
        break
