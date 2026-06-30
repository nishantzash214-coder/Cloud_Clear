import requests
import pathlib

p = pathlib.Path('data/raw/optical/test_scene.tif')
print('Exists', p.exists())
with p.open('rb') as f:
    r = requests.post('http://127.0.0.1:8000/infer', files={'optical': ('test_scene.tif', f, 'image/tiff')})
    print('status', r.status_code)
    print(r.text)
