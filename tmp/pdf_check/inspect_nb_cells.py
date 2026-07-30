import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if 'generate_dataset' in src or 'np.savez_compressed' in src or 'split_dataset' in src or 'GroupShuffleSplit' in src or 'y_type_list.append' in src or 'y_loc_list.append' in src:
            print(f'CELL {i}:')
            print(src[:3500])
            print('---')
