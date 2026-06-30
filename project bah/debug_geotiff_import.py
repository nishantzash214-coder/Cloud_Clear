import sys
from pathlib import Path
print('cwd=', Path('.').resolve())
print('sys.path[0]=', sys.path[0])
print('sys.path contains project root?', str(Path('.').resolve()) in sys.path)
try:
    import geotiff
    print('geotiff imported', geotiff.__file__)
    print('has read_geotiff', hasattr(geotiff,'read_geotiff'))
except Exception as e:
    print('import geotiff failed', repr(e))
try:
    from src.utils.geotiff import read_geotiff
    print('import src.utils.geotiff ok', read_geotiff)
except Exception as e:
    print('import src.utils.geotiff failed', repr(e))
