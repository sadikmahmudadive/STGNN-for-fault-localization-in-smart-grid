import sys
from pathlib import Path
sys.path.insert(0, str(Path('tmp/pydeps').resolve()))
from pypdf import PdfReader
from pypdf.generic import IndirectObject
pdf = Path(r"Reports/Graph_Neural_Networks__GNN__for_Fault_Detection_and_Localization_in_Power_Grids.pdf")
reader = PdfReader(str(pdf))
for i, page in enumerate(reader.pages, start=1):
    images = 0
    try:
        resources = page.get('/Resources')
        if isinstance(resources, IndirectObject):
            resources = resources.get_object()
        xobj = resources.get('/XObject') if resources else None
        if isinstance(xobj, IndirectObject):
            xobj = xobj.get_object()
        if xobj:
            for name, obj in xobj.items():
                if isinstance(obj, IndirectObject):
                    obj = obj.get_object()
                subtype = obj.get('/Subtype')
                if str(subtype) == '/Image':
                    images += 1
                elif str(subtype) == '/Form':
                    # count nested image xobjects inside forms
                    nested = obj.get('/Resources', {}).get('/XObject')
                    if nested:
                        if isinstance(nested, IndirectObject):
                            nested = nested.get_object()
                        for _, nobj in nested.items():
                            if isinstance(nobj, IndirectObject):
                                nobj = nobj.get_object()
                            if str(nobj.get('/Subtype')) == '/Image':
                                images += 1
    except Exception as e:
        print(f'PAGE {i}: error {e}')
        continue
    print(f'PAGE {i}: image_objects={images}')
