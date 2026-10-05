# -*- coding: utf-8 -*-
"""
產生 GRTv3_ada.py 的流程圖(APA 7 格式的圖像本體)。

    python review/fig/make_fig.py            -> review/fig/GRTv3_ada_trial_flow.svg

轉 PNG(任選其一):
    chromium --headless --screenshot=GRTv3_ada_trial_flow.png \
             --window-size=2120,1480 GRTv3_ada_trial_flow.svg
    inkscape GRTv3_ada_trial_flow.svg --export-type=png --export-dpi=300

APA 7 §7.22–7.28 的限制,也是這張圖為什麼長這樣:
  - 圖內只放面板字母(A–D)、單詞標籤、時間、座標軸與圖例;
    解釋性文字一律寫在圖下方的 Note(見 GRTv3_ada_trial_flow_note.md)。
  - 圖內字體用無襯線 8–14 pt(§7.26);Figure 編號、標題、Note 用內文字體。
  - 不依賴 matplotlib / PIL:PsychoPy 的 Python 沒有它們,純手寫 SVG 到哪都能跑。

幾個刻意的選擇:
  - 螢幕底色 #7f7f7f:PsychoPy rgb [0,0,0] 是中灰,不是黑。
  - 兩個刺激顏色取 agrt_colour_lut.json 中 COLOUR_ARC = [-3, +3] 的查表值,
    所以看起來幾乎一樣 —— 實驗裡就是這樣,不要為了「好看」改成藍/粉。
  - 作答畫面中央放 stimuli/box.png,與 GRTv3_ada.py 的 BOX ImageStim 相同。
  - D 右側的收斂曲線是示意(固定亂數種子),不是受試者資料;Note 要說明。
"""
import base64
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parents[2]       # repo 根目錄
OUT  = pathlib.Path(__file__).with_name("GRTv3_ada_trial_flow.svg")

W, H = 2120, 1480
FONT   = "Arial, Helvetica, Liberation Sans, sans-serif"   # APA §7.26:圖內無襯線
C1, C2 = "#827CBE", "#9477B7"       # LUT arc -3 / +3 dE00
ANCHOR = "#8B7ABB"                  # arc 0 = 錨點色(校準聲音時四個項目都是它)
SCR_BG = "#7f7f7f"                  # PsychoPy rgb [0,0,0]
PW, PH = 196, 147                   # 螢幕面板尺寸(px)
BOX_B64 = base64.b64encode((ROOT / "stimuli" / "box.png").read_bytes()).decode()

out = []
def add(s): out.append(s)

def text(x, y, s, size=12, anchor="middle", weight="normal", fill="#222", extra=""):
    add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}" font-family="{FONT}" {extra}>{s}</text>')

def speaker(cx, cy, noisy=False, scale=1.0, color="white"):
    """喇叭圖示:弧線 = 高 SNR,鋸齒 = 低 SNR(圖例會解釋)。"""
    add(f'<g transform="translate({cx},{cy}) scale({scale})">'
        f'<path d="M-9,-4 h5 l6,-5 v18 l-6,-5 h-5 z" fill="{color}"/>')
    if noisy:
        add(f'<path d="M5,-7 l3,3 l-3,3 l3,3 l-3,3 l3,3 l-3,3" stroke="{color}" stroke-width="1.6" fill="none"/>'
            f'<path d="M10,-9 l3,3 l-3,3 l3,3 l-3,3 l3,3 l-3,3 l3,3" stroke="{color}" stroke-width="1.6" fill="none" opacity="0.7"/>')
    else:
        add(f'<path d="M5,-4 a6,6 0 0 1 0,8" stroke="{color}" stroke-width="1.8" fill="none"/>'
            f'<path d="M8,-8 a11,11 0 0 1 0,16" stroke="{color}" stroke-width="1.8" fill="none"/>')
    add('</g>')

def panel(x, y, w=PW, h=PH):
    add(f'<rect x="{x+4}" y="{y+5}" width="{w}" height="{h}" rx="4" fill="#000" opacity="0.18"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{SCR_BG}" stroke="#333" stroke-width="1.6"/>')

CORN = {"UL": (-54, -40), "UR": (54, -40), "BL": (-54, 40), "BR": (54, 40)}
def cpos(px, py, corner):
    return px + PW/2 + CORN[corner][0], py + PH/2 + CORN[corner][1]

def square(cx, cy, color, size=30):
    add(f'<rect x="{cx-size/2}" y="{cy-size/2}" width="{size}" height="{size}" fill="{color}"/>')

def study_item(px, py, corner, color, noisy):
    cx, cy = cpos(px, py, corner)
    square(cx, cy, color)
    dx = 1 if corner.endswith("L") else -1        # 喇叭畫在內側,才不會被面板邊切掉
    speaker(cx + dx*31, cy, noisy=noisy, scale=0.9)

def cross(px, py):
    cx, cy = px + PW/2, py + PH/2
    add(f'<path d="M{cx-7},{cy} h14 M{cx},{cy-7} v14" stroke="white" stroke-width="2"/>')

def frame(px, py, corner, size=44):
    cx, cy = cpos(px, py, corner)
    add(f'<rect x="{cx-size/2}" y="{cy-size/2}" width="{size}" height="{size}" fill="none" stroke="white" stroke-width="2.2"/>')

def option(px, py, corner, color, label):
    cx, cy = cpos(px, py, corner)
    square(cx, cy + 6, color, size=24)
    text(cx, cy - 11, label, size=11, fill="white", weight="bold")

def cedrus(px, py):
    """作答畫面中央的 Cedrus 圖(stimuli/box.png)。"""
    cx, cy = px + PW/2, py + PH/2
    w, h = 56, 46
    add(f'<image x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" '
        f'href="data:image/png;base64,{BOX_B64}" preserveAspectRatio="xMidYMid meet"/>')

def caption(px, py, title, dur):
    text(px + PW/2, py + PH + 22, title, size=13, weight="bold")
    text(px + PW/2, py + PH + 40, dur, size=12, fill="#444")

def panel_letter(x, y, letter):
    text(x, y, letter, size=20, anchor="start", weight="bold")

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add(f'<rect width="{W}" height="{H}" fill="white"/>')
add('<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>'
    '<marker id="arrg" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#999"/></marker></defs>')

# ───────── A ─────────
panel_letter(40, 40, "A")
stages = [("Instructions", "", "#eeeeee", 150),
          ("Colour calibration", "72 trials", "#dfe6f5", 230),
          ("Sound calibration", "72 trials", "#e8dff2", 230),
          ("Practice", "15 trials", "#f3ead9", 170),
          ("Main experiment", "4 blocks × 144 trials", "#dcefe0", 280),
          ("End", "", "#eeeeee", 110)]
x, y, h = 40, 60, 64
for i, (t, sub, col, w) in enumerate(stages):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{col}" stroke="#555" stroke-width="1.2"/>')
    text(x + w/2, y + (28 if not sub else 24), t, size=13, weight="bold")
    if sub: text(x + w/2, y + 44, sub, size=11, fill="#444")
    if i < len(stages) - 1:
        add(f'<line x1="{x+w+2}" y1="{y+h/2}" x2="{x+w+26}" y2="{y+h/2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
        if i >= 1: text(x + w + 14, y + h/2 - 8, "rest", size=9, fill="#666")
    x += w + 28

# ───────── B:單一 invalid 試次 ─────────
panel_letter(40, 185, "B")
X0, Y0 = 40, 210
DX, DY = 218, 70
def P(i): return X0 + i*DX, Y0 + i*DY

px, py = P(0); panel(px, py); cross(px, py); caption(px, py, "Fixation", "0.3 s")

# item 編碼同 GRTv3_ada.py:0 = C1 clear [bi], 1 = C2 clear [bi], 2 = C1 noisy [pi], 3 = C2 noisy [pi]
study = [("UL", C2, False), ("BR", C1, True), ("UR", C1, False), ("BL", C2, True)]
for k, (corner, col, noisy) in enumerate(study):
    px, py = P(1 + k); panel(px, py); study_item(px, py, corner, col, noisy)   # 注視點只在前 0.3 s
    caption(px, py, f"Study item {k+1}", "1 s")

px, py = P(5); panel(px, py); frame(px, py, "BR")              # 框在 cued item(item 2)的位置
caption(px, py, "Cue", "1 s")

px, py = P(6); panel(px, py); frame(px, py, "BR")              # 內容是 target(item 3)-> invalid
cx, cy = cpos(px, py, "BR"); square(cx, cy, C2, size=30); speaker(cx - 33, cy, noisy=True, scale=0.9)
caption(px, py, "Probe", "1 s")

px, py = P(7); panel(px, py)
opts = [("UL", C1, "[pi]"), ("UR", C2, "[bi]"), ("BL", C1, "[bi]"), ("BR", C2, "[pi]")]
for corner, col, lab in opts: option(px, py, corner, col, lab)
cedrus(px, py)
caption(px, py, "Response", "until button press")
add(f'<line x1="{px+PW+12}" y1="{py+PH/2}" x2="{px+PW+60}" y2="{py+PH/2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
text(px+PW+66, py+PH/2 + 4, "Next trial", size=12, anchor="start")

ax0, ay0 = X0 - 10, Y0 + PH + 70
ax1, ay1 = P(7)[0] - 10, P(7)[1] + PH + 70
add(f'<line x1="{ax0}" y1="{ay0}" x2="{ax1}" y2="{ay1}" stroke="#999" stroke-width="2" marker-end="url(#arrg)"/>')
text(ax1 - 30, ay1 + 22, "Time", size=12, fill="#777", anchor="end")

# 圖例(APA 允許放圖內:只解釋符號)
lx, ly = X0, Y0 + PH + 210
add(f'<rect x="{lx}" y="{ly}" width="200" height="62" fill="white" stroke="#999" stroke-width="1"/>')
add(f'<rect x="{lx+8}" y="{ly+8}" width="184" height="46" fill="{SCR_BG}"/>')
speaker(lx+30, ly+22, noisy=False, scale=0.8); text(lx+48, ly+26, "high SNR, [bi]", size=11, anchor="start", fill="white")
speaker(lx+30, ly+42, noisy=True,  scale=0.8); text(lx+48, ly+46, "low SNR, [pi]",  size=11, anchor="start", fill="white")

# ───────── C:時間軸 ─────────
ty = Y0 + 7*DY + PH + 110
panel_letter(40, ty, "C")
ty += 20
segs = [("Fixation", 0.3, "#cccccc"), ("Study", 4.0, "#dfe6f5"), ("Cue", 1.0, "#eeeeee"),
        ("Probe", 1.0, "#e8dff2"), ("Response", 1.6, "#f3ead9")]
scale, x = 110, 40
for name, dur, col in segs:
    w = dur * scale
    add(f'<rect x="{x}" y="{ty}" width="{w}" height="36" fill="{col}" stroke="#444" stroke-width="1.2"/>')
    text(x + w/2, ty + 23, name if dur > 0.5 else "", size=12)
    text(x + w/2, ty + 52, f"{dur:g} s" if name != "Response" else "RT", size=11, fill="#555")
    x += w

# ───────── D:聲音校準(Psi) ─────────
dy0 = ty + 95
panel_letter(40, dy0, "D")
dy0 += 60
sx, sy = 40, dy0
panel(sx, sy)
for corner, noisy in [("UL", False), ("UR", True), ("BL", True), ("BR", False)]:
    study_item(sx, sy, corner, ANCHOR, noisy)
for corner, sgn in [("UL", "+|s| dB"), ("UR", "−|s| dB"), ("BL", "−|s| dB"), ("BR", "+|s| dB")]:
    cx, cy = cpos(sx, sy, corner); text(cx, cy + 30, sgn, size=10, fill="white")
caption(sx, sy, "Study", "SNR = ±|s|")
add(f'<line x1="{sx+PW+10}" y1="{sy+PH/2}" x2="{sx+PW+50}" y2="{sy+PH/2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')

rx, ry = sx + PW + 60, sy
panel(rx, ry)
for corner, lab in [("UL", "[bi]"), ("UR", "[pi]"), ("BL", "[pi]"), ("BR", "[bi]")]:
    option(rx, ry, corner, ANCHOR, lab)
cedrus(rx, ry)
caption(rx, ry, "Response", "[bi] or [pi]")

bx = rx + PW + 60
add(f'<line x1="{rx+PW+10}" y1="{ry+PH/2}" x2="{bx-8}" y2="{ry+PH/2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
add(f'<rect x="{bx}" y="{ry+PH/2-34}" width="150" height="68" rx="8" fill="#fff4d6" stroke="#555" stroke-width="1.4"/>')
text(bx+75, ry+PH/2+5, "Psi", size=14, weight="bold")
loop_y = sy - 30
add(f'<path d="M{bx+75},{ry+PH/2-34} V{loop_y} H{sx+PW/2} V{sy-6}" fill="none" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
text((bx+75+sx+PW/2)/2, loop_y - 6, "× 72 trials", size=11, fill="#444")

# 示意的收斂圖(固定種子,非資料)
gx, gy, gw, gh = bx + 200, sy - 10, 560, 200
add(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" fill="#fafafa" stroke="#999" stroke-width="1"/>')
def Y(db): return gy + gh/2 - db/18*(gh/2)
def Xn(n): return gx + 20 + (n-1)/71*(gw-40)
add(f'<line x1="{gx}" y1="{Y(0)}" x2="{gx+gw}" y2="{Y(0)}" stroke="#bbb" stroke-width="1" stroke-dasharray="3,3"/>')
for db in (18, 0, -18):
    text(gx - 6, Y(db) + 4, f"{db:+d}" if db else "0", size=10, anchor="end", fill="#555")
text(gx - 28, gy + gh/2, "SNR (dB)", size=11, fill="#555", extra=f'transform="rotate(-90 {gx-28} {gy+gh/2})"')
text(gx + gw/2, gy + gh + 26, "Trial", size=11, fill="#555")
for n in (1, 36, 72): text(Xn(n), gy + gh + 13, str(n), size=10, fill="#555")
rnd = random.Random(7)
pts = []
for n in range(1, 73):
    mag = 2.2 + 14 * math.exp(-n / 14) + rnd.uniform(-0.8, 0.8)
    s = mag if rnd.random() < 0.5 else -mag
    pts.append((Xn(n), Y(s)))
add('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '" fill="none" stroke="#999" stroke-width="1"/>')
for x, y in pts: add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#5b4a9c"/>')
for db, lab, col in ((4.5, "SNR hi", C1), (-4.5, "SNR lo", C2)):
    add(f'<line x1="{gx}" y1="{Y(db)}" x2="{gx+gw}" y2="{Y(db)}" stroke="{col}" stroke-width="2" stroke-dasharray="8,4"/>')
    text(gx + gw + 8, Y(db) + 4, lab, size=11, anchor="start", fill="#333", weight="bold")
add(f'<line x1="{gx+gw+70}" y1="{gy+gh/2}" x2="{gx+gw+118}" y2="{gy+gh/2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
text(gx+gw+124, gy+gh/2 - 6, "Practice and", size=12, anchor="start")
text(gx+gw+124, gy+gh/2 + 10, "main experiment", size=12, anchor="start")

add('</svg>')
OUT.write_text("\n".join(out), encoding="utf-8")
print(OUT)
