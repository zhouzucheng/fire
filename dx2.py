import rasterio
import numpy as np


file1_path='../Data/dx1.tif'
file2_path='../Data/dx2.tif'

# 打开第一个栅格文件和第二个栅格文件
with rasterio.open(file1_path) as raster1, rasterio.open(file2_path) as raster2:
    # 读取栅格数据
    data1 = raster1.read(1).astype(np.int8)  # 确保数据类型为int8
    data2 = raster2.read(1).astype(np.int8)  # 确保数据类型为int8

    # 确保两个栅格文件的形状相同
    if data1.shape != data2.shape:
        raise ValueError("两个栅格文件的形状不相同")

    # 计算栅格值的差异
    difference = np.subtract(data1, data2)  # 使用numpy的subtract函数进行安全计算

    # 设置新栅格文件的元数据
    new_raster_meta = raster1.meta.copy()
    new_raster_meta.update(dtype=rasterio.int8)  # 更新数据类型为int8以支持-1, 0, 1的值

    # 写入新的栅格文件
    with rasterio.open('../Data/dx3.tif', 'w', **new_raster_meta) as new_raster:
        new_raster.write(difference, 1)

print("新的栅格文件已生成：difference_result.tif")
