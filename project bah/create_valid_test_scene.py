import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

Path('data/raw/optical').mkdir(parents=True, exist_ok=True)

# 4-band image 512x512
array = np.zeros((4, 512, 512), dtype='float32')
array[0, :256, :256] = 0.08
array[1, :256, :256] = 0.06
array[2, :256, :256] = 0.55
array[3, :256, :256] = 0.22
array[0, :256, 256:] = 0.06
array[1, :256, 256:] = 0.04
array[2, :256, 256:] = 0.04
array[3, :256, 256:] = 0.02
array[0, 256:, :256] = 0.22
array[1, 256:, :256] = 0.20
array[2, 256:, :256] = 0.28
array[3, 256:, :256] = 0.30
array[0, 256:, 256:] = 0.25
array[1, 256:, 256:] = 0.22
array[2, 256:, 256:] = 0.30
array[3, 256:, 256:] = 0.28
array[:, 100:200, 100:300] = 0.8

transform = from_bounds(77.2, 23.1, 77.6, 23.5, 512, 512)
profile = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'count': 4,
    'height': 512,
    'width': 512,
    'crs': 'EPSG:4326',
    'transform': transform,
    'compress': 'lzw',
}
with rasterio.open('data/raw/optical/test_scene.tif', 'w', **profile) as dst:
    dst.write(array)

print('Created valid GeoTIFF at data/raw/optical/test_scene.tif')
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

Path('data/raw/optical').mkdir(parents=True, exist_ok=True)

# Create a valid 4-band float32 GeoTIFF
array = np.zeros((4, 512, 512), dtype='float32')
array[0, :256, :256] = 0.08
array[1, :256, :256] = 0.06
array[2, :256, :256] = 0.55
array[3, :256, :256] = 0.22

array[0, :256, 256:] = 0.06
array[1, :256, 256:] = 0.04
array[2, :256, 256:] = 0.04
array[3, :256, 256:] = 0.02

array[0, 256:, :256] = 0.22
array[1, 256:, :256] = 0.20
array[2, 256:, :256] = 0.28
array[3, 256:, :256] = 0.30

array[0, 256:, 256:] = 0.25
array[1, 256:, 256:] = 0.22
array[2, 256:, 256:] = 0.30
array[3, 256:, 256:] = 0.28

array[:, 100:200, 100:300] = 0.82

transform = from_bounds(77.2, 23.1, 77.6, 23.5, 512, 512)
meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'count': 4,
    'width': 512,
    'height': 512,
    'crs': 'EPSG:4326',
    'transform': transform,
}

with rasterio.open('data/raw/optical/test_scene.tif', 'w', **meta) as dst:
    dst.write(array)

print('Created valid GeoTIFF: data/raw/optical/test_scene.tif')
