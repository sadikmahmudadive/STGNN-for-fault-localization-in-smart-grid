import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
nb = json.loads(Path('ST_GNN_Full_Pipeline.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if any(k in src for k in ['group', 'fault_bus', 'scenario', 'bus_idx', 'bus_id', 'load_scale', 'fault_res', 'y_loc_raw', 'TARGET_FAULT_BUS']):
            print(f'CELL {i}:')
            print(src[:2500])
            print('---')
