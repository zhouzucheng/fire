import rasterio
import numpy as np


with rasterio.open('../Data/zzz_lssvm.tif') as predict_src:
    predict_data = predict_src.read(1)
    profile = predict_src.profile


with rasterio.open('../Data/flipped_raster.tif') as burned_src:
    burned_data = burned_src.read(1)


dx1_data = np.where(predict_data == burned_data, 1, 0)


profile.update(dtype=rasterio.uint8, count=1)


with rasterio.open('../Data/dx2.tif', 'w', **profile) as dst:
    dst.write(dx1_data.astype(rasterio.uint8), 1)