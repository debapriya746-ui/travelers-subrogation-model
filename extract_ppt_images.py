import re
import zipfile
from pathlib import Path

ppt_path = Path(r'D:\UCONN\Travellers modeling competetion\2025 Travelers University Modeling Competition [Autosaved].pptx')
out_dir = Path(r'D:\UCONN\Travellers modeling competetion\ppt_images')
out_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ppt_path) as z:
    image_names = [
        name for name in z.namelist()
        if name.startswith('ppt/media/') and re.search(r'\.(png|jpg|jpeg|gif|bmp|tif|tiff|svg)$', name, re.I)
    ]

    if not image_names:
        print('NO_MEDIA_FOUND')
    else:
        for idx, name in enumerate(image_names, 1):
            ext = Path(name).suffix or '.png'
            dest = out_dir / f'image_{idx:03d}{ext}'
            with z.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            print(f'WROTE {dest.name}')

        print('OUTPUT_DIR', out_dir)
        for p in sorted(out_dir.glob('*')):
            print(p.name, p.stat().st_size)
