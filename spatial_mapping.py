import matplotlib
matplotlib.use('Agg')  # 非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 與 create_impulse.py 相同的參數
az_values   = [-45, -35, -20, -10, 0, 10, 20, 35, 45]   # 左到右
dist_values = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]          # 近到遠
el_values   = [(0, -45), (8, 0), (16, 45)]              # (CIPIC索引, 角度)

# 建立對照表
mapping = []
n = 1
for el_idx, el_deg in el_values:
    for dist in dist_values:
        for az in az_values:
            mapping.append({
                'stimulus': n,
                'azimuth': az,
                'elevation': el_deg,
                'distance': dist
            })
            n += 1

df = pd.DataFrame(mapping)
df.to_csv('/mnt/c/Users/spt904/Desktop/stimuli/spatial_mapping.csv', index=False)
print("已儲存 spatial_mapping.csv")

# ============ 視覺化 ============
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (_, el_deg) in enumerate(el_values):
    ax = axes[i]
    subset = df[df['elevation'] == el_deg]

    # 建立 9x9 網格
    grid = np.zeros((9, 9), dtype=int)
    for _, row in subset.iterrows():
        az_idx = az_values.index(row['azimuth'])
        dist_idx = dist_values.index(row['distance'])
        grid[dist_idx, az_idx] = row['stimulus']

    # 繪製熱力圖
    im = ax.imshow(grid, cmap='viridis', aspect='auto')

    # 標記每個格子的 stimulus 編號
    for y in range(9):
        for x in range(9):
            ax.text(x, y, str(grid[y, x]), ha='center', va='center',
                   color='white', fontsize=8, fontweight='bold')

    ax.set_xticks(range(9))
    ax.set_xticklabels([f'{a}°' for a in az_values])
    ax.set_yticks(range(9))
    ax.set_yticklabels([f'{d}m' for d in dist_values])
    ax.set_xlabel('Azimuth (左 ← → 右)')
    ax.set_ylabel('Distance (近 ↑ ↓ 遠)')
    ax.set_title(f'Elevation = {el_deg}°\n({"下方" if el_deg < 0 else "水平" if el_deg == 0 else "上方"})')

plt.suptitle('Stimulus Spatial Mapping\n(stimulus 編號對照空間位置)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/c/Users/spt904/Desktop/stimuli/spatial_mapping.png', dpi=150)
# plt.show()  # 在無顯示環境中跳過
print("已儲存 spatial_mapping.png")

# ============ 文字對照表 ============
print("\n" + "="*60)
print("STIMULUS 空間對照表")
print("="*60)
for _, el_deg in el_values:
    print(f"\n【Elevation = {el_deg}° ({'下方' if el_deg < 0 else '水平' if el_deg == 0 else '上方'})】")
    print(f"{'Distance':>8} | " + " | ".join([f'{a:>4}°' for a in az_values]))
    print("-" * 60)
    subset = df[df['elevation'] == el_deg]
    for dist in dist_values:
        row_data = subset[subset['distance'] == dist]['stimulus'].values
        print(f"{dist:>6}m  | " + " | ".join([f'{s:>4}' for s in row_data]))
