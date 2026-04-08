import rasterio
import numpy as np

# 读取 predict 栅格
with rasterio.open('../Data/zzz_lssvm.tif') as predict_src:
    predict_data = predict_src.read(1)  # 读取第一个波段数据
    profile = predict_src.profile  # 复制栅格的元数据

# 读取 burned 栅格
with rasterio.open('../Data/flipped_raster.tif') as burned_src:
    burned_data = burned_src.read(1)  # 读取第一个波段数据

# 生成 dx1 栅格
dx1_data = np.where(predict_data == burned_data, 1, 0)

# 更新元数据
profile.update(dtype=rasterio.uint8, count=1)

# 保存 dx1 栅格
with rasterio.open('../Data/dx2.tif', 'w', **profile) as dst:
    dst.write(dx1_data.astype(rasterio.uint8), 1)