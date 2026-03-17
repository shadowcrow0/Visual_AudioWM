import numpy as np

# # 音源在正左方 [-2, 0, 0]
# x, y, z = -2, 0, 0
# az = np.degrees(np.arctan2(x, y))
# print("左方音源的 azimuth:", az)  # 應該是 -90

# # 音源在正右方 [2, 0, 0]
# x, y, z = 2, 0, 0
# az = np.degrees(np.arctan2(x, y))
# print("右方音源的 azimuth:", az)  # 應該是 90


# import netCDF4

# f = netCDF4.Dataset('/home/yyc/H3_48K_24bit_256tap_FIR_SOFA.sofa', 'r')
# pos = f.variables['SourcePosition'][:]
# f.close()

# # 找 azimuth 接近 90 的點
# az = pos[:, 0]
# idx = np.argmin(np.abs(az - 90))
# print("az=90 的完整座標:", pos[idx])

# # 找 azimuth 接近 -90 的點
# idx = np.argmin(np.abs(az - (-90)))
# print("az=-90 的完整座標:", pos[idx])


import netCDF4
import numpy as np

f = netCDF4.Dataset('/home/yyc/H5_96K_24bit_512tap_FIR_SOFA.sofa', 'r')
pos = f.variables['SourcePosition'][:]
f.close()

az_unique = np.unique(pos[:, 0])
print("所有 azimuth 值:", az_unique)