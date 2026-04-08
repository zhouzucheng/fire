import rasterio
import numpy as np

with rasterio.open('../Data/zzz_burned.tif') as src:
    data = src.read(1)
    profile = src.profile


flipped_data = np.flipud(data)


profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open('../Data/flipped_raster.tif', 'w', **profile) as dst:
    dst.write(flipped_data.astype(rasterio.uint8), 1)