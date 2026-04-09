import numpy as np
from skimage import color as skcolor

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def delta_e(c1, c2):
    return float(np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2)))

def is_in_gamut(L, a, b):
    rgb = skcolor.lab2rgb(np.array([[[L, a, b]]]))[0][0]
    return bool(np.all(rgb >= 0) and np.all(rgb <= 1))

def lab_to_hex(lab):
    L, a, b = lab
    rgb = skcolor.lab2rgb(np.array([[[L, a, b]]]))[0][0]
    r, g, b_ch = [int(x * 255) for x in np.clip(rgb, 0, 1)]
    return '#{:02X}{:02X}{:02X}'.format(r, g, b_ch)

def random_lab():
    L     = np.random.uniform(40, 60)
    C     = np.random.uniform(40, 60)
    H_rad = np.random.uniform(0, 2 * np.pi)
    return np.array([L, C * np.cos(H_rad), C * np.sin(H_rad)])

def find_color(target, de_min, de_max, max_tries=2000):
    for _ in range(max_tries):
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        candidate = target + direction * np.random.uniform(de_min, de_max)
        L, a, b = candidate
        if is_in_gamut(L, a, b) and de_min < delta_e(target, candidate) < de_max:
            return candidate
    return None

# ------------------------------------------------------------
# Generate color pool
# ------------------------------------------------------------

color_pool = []
all_targets = []

while len(color_pool) < 24:

    # target1: random valid color
    t1 = None
    for _ in range(500):
        c = random_lab()
        if is_in_gamut(*c):
            t1 = c
            break
    if t1 is None:
        continue

    # target2: ΔE > 20 from target1
    t2 = find_color(t1, 20, 120)
    if t2 is None:
        continue

    # check both targets are far from all previous targets
    if any(delta_e(t1, prev) < 20 or delta_e(t2, prev) < 20 for prev in all_targets):
        continue

    # high1: ΔE > 20 from target1
    h1 = find_color(t1, 20, 80)
    if h1 is None:
        continue

    # low1: 2 < ΔE < 10 from target1
    l1 = find_color(t1, 2, 10)
    if l1 is None:
        continue

    # high2: ΔE > 20 from target2
    h2 = find_color(t2, 20, 80)
    if h2 is None:
        continue

    # low2: 2 < ΔE < 10 from target2
    l2 = find_color(t2, 2, 10)
    if l2 is None:
        continue

    # all conditions met, add to pool
    color_pool.append({
        'target1': lab_to_hex(t1),
        'high1':   lab_to_hex(h1),
        'low1':    lab_to_hex(l1),
        'target2': lab_to_hex(t2),
        'high2':   lab_to_hex(h2),
        'low2':    lab_to_hex(l2),
    })
    all_targets.extend([t1, t2])
    print(f"Entry {len(color_pool):2d}/24  t1={lab_to_hex(t1)}  t2={lab_to_hex(t2)}")

print("Done.")