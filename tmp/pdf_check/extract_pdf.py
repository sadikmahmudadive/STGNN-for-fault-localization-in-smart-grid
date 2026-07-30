import sys
from pathlib import Path
sys.path.insert(0, str(Path('tmp/pydeps').resolve()))
from pypdf import PdfReader
pdf = Path(r"Reports/Graph_Neural_Networks__GNN__for_Fault_Detection_and_Localization_in_Power_Grids.pdf")
reader = PdfReader(str(pdf))
out = Path('tmp/pdf_check')
out.mkdir(parents=True, exist_ok=True)
text_path = out/'extracted.txt'
with text_path.open('w', encoding='utf-8') as f:
    f.write(f'PAGES: {len(reader.pages)}\n\n')
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ''
        f.write(f'===== PAGE {i} =====\n')
        f.write(text)
        f.write('\n\n')
print(text_path)
print(f'PAGES={len(reader.pages)}')
for i, page in enumerate(reader.pages, start=1):
    text = (page.extract_text() or '').replace('\n',' ')
    print(f'PAGE_{i}_HEAD=' + text[:500])
