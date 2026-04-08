import rasterio
import numpy as np


file1_path='../Data/dx1.tif'
file2_path='../Data/dx2.tif'

with rasterio.open(file1_path) as raster1, rasterio.open(file2_path) as raster2:

    data1 = raster1.read(1).astype(np.int8)
    data2 = raster2.read(1).astype(np.int8)


    if data1.shape != data2.shape:
        raise ValueError("两个栅格文件的形状不相同")


    difference = np.subtract(data1, data2)

    new_raster_meta = raster1.meta.copy()
    new_raster_meta.update(dtype=rasterio.int8)


    with rasterio.open('../Data/dx3.tif', 'w', **new_raster_meta) as new_raster:
        new_raster.write(difference, 1)

