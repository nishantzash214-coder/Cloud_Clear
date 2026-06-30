import numpy as np
from pathlib import Path

# Create directory
Path('data/raw/optical').mkdir(parents=True, exist_ok=True)

# Create simple test data (4-band image, 512x512)
data = np.random.rand(4, 512, 512).astype('float32') * 0.3
# Add some synthetic cloud
data[:, 100:200, 100:200] = 0.8

# Save as raw binary (can be read as GeoTIFF later)
data.tofile('data/raw/optical/test_scene.dat')

print('Test data created at: data/raw/optical/test_scene.dat')
print(f'Shape: {data.shape}')
