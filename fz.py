import rasterio
import numpy as np

# 读取原始 TIF 文件
with rasterio.open('../Data/zzz_burned.tif') as src:
    data = src.read(1)  # 读取第一个波段数据
    profile = src.profile  # 获取栅格的元数据

# 使用 np.flipud() 对数据进行 x 轴翻转
flipped_data = np.flipud(data)

# 更新元数据（数据类型和波段数等）
profile.update(dtype=rasterio.uint8, count=1)

# 保存翻转后的数据为新的 TIF 文件
with rasterio.open('../Data/flipped_raster.tif', 'w', **profile) as dst:
    dst.write(flipped_data.astype(rasterio.uint8), 1)

print("栅格文件已成功翻转并保存为 flipped_raster.tif。")
