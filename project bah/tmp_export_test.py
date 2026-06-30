import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from exporter import SceneExporter

os.chdir(Path(__file__).parent)
print('cwd', Path.cwd())
print('sys.path[0]', Path.cwd())

cfg = OmegaConf.load('base.yaml')
exporter = SceneExporter(cfg, Path('tmp_export_test'))
recon = torch.rand(1, 4, 8, 8)
mask = torch.zeros(1, 8, 8, dtype=torch.long)

class NS:
    pass

conf_maps = NS()
conf_maps.confidence = torch.rand(1, 1, 8, 8)
conf_maps.uncertainty = torch.zeros(1, 1, 8, 8, dtype=torch.int8)
conf_maps.mc_variance = None
conf_maps.breakdown = {}

report = {'overall_pass': True, 'passes': []}
paths = exporter.export(
    'scene_test', recon, mask, conf_maps,
    {'driver': 'GTiff', 'crs': 'EPSG:32643', 'transform': None},
    report
)
print('paths:', paths)
print('path types:', {k: type(v) for k, v in paths.items()})
