import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

Path('data/raw/optical').mkdir(parents=True, exist_ok=True)

data = np.zeros((4, 512, 512), dtype='float32')
data[0, :256, :256] = 0.08
data[1, :256, :256] = 0.06
data[2, :256, :256] = 0.55
data[3, :256, :256] = 0.22

data[0, :256, 256:] = 0.06
data[1, :256, 256:] = 0.04
data[2, :256, 256:] = 0.04
data[3, :256, 256:] = 0.02

data[0, 256:, :256] = 0.22
data[1, 256:, :256] = 0.20
data[2, 256:, :256] = 0.28
data[3, 256:, :256] = 0.30

data[0, 256:, 256:] = 0.25
data[1, 256:, 256:] = 0.22
data[2, 256:, 256:] = 0.30
data[3, 256:, 256:] = 0.28

# Add synthetic cloud patch
data[0, 100:200, 100:300] = 0.82
data[1, 100:200, 100:300] = 0.80
data[2, 100:200, 100:300] = 0.78
data[3, 100:200, 100:300] = 0.65

transform = from_bounds(77.2, 23.1, 77.6, 23.5, 512, 512)
with rasterio.open(
    'data/raw/optical/test_scene.tif', 'w',
    driver='GTiff', count=4, dtype='float32',
    crs='EPSG:4326', transform=transform,
    width=512, height=512,
) as dst:
    dst.write(data)

print('Test GeoTIFF created at: data/raw/optical/test_scene.tif')
