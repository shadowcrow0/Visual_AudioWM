import netCDF4
import numpy as np


# 找 az=0, el=0 附近的點，看它實際對應哪個方向
f = netCDF4.Dataset('/home/yyc/H5_48K_24bit_256tap_FIR_SOFA.sofa', 'r')
print("座標類型:", f.variables['SourcePosition'].Type)
print("座標單位:", f.variables['SourcePosition'].Units)

pos = f.variables['SourcePosition'][:]
# 找最接近 az=0, el=0 的點
idx = np.argmin(np.abs(pos[:,0]) + np.abs(pos[:,1]))
print("az=0, el=0 附近的點:", pos[idx])

# 也看看 ReceiverPosition
print("ReceiverPosition:", f.variables['ReceiverPosition'][:])
f.close()