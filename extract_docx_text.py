import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

path = Path(r'D:\UCONN\Travellers modeling competetion\2025UMC_Business_Problem (3).docx')

with zipfile.ZipFile(path) as z:
    text_parts = []
    for name in z.namelist():
        if not name.endswith('.xml'):
            continue
        data = z.read(name)
        if b'<w:t' not in data:
            continue
        root = ET.fromstring(data)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        for node in root.findall('.//w:t', ns):
            if node.text:
                text_parts.append(node.text)

print('\n'.join(text_parts[:4000]))
