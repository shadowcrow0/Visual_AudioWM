# GRT 擬合 M&N：B 路線 D1 紀錄

2026-11-10。對應 `企劃_GRT擬合MillerNicely.md` §4b。軸用 A4（voicing、nasality、三階 manner、place）。

## 做了什麼

```
miller_nicely_1955_tables_I-VI.json (16×16 × 4 SNR)
        │
        v
grt_mn_subdesigns.py ── 枚舉極大全因子子設計 (11 個) ──> review/data/grt_subdesigns/
        │                                               <design>_<snr>.csv  長格式 (44 檔)
        │                                               index.csv
        └── 簡單 SDT：邊際 d′（整體／分另一軸）+ MRI χ²  ──> python_sdt_summary.{txt,csv}
                                                                │
grt_mn_subdesigns.R ── 讀 index.csv → mdsdt::fit.grt ──────────┘  (筆電上跑)
                       → r_mdsdt_results.csv、r_mdsdt_<design>.pdf
tests/test_grt_mn_subdesigns.py  7 tests 通過
```

子設計 = 固定另外兩軸後，兩軸水準的極大子集且每格都有子音；同一脈絡可有兩個（manner × place）。
所有數字**條件化在子集內的反應**上（`index.csv` 的 `prop_retained`，−6 dB 約 .55–.80，p/t/k 低於 .5）。

## 子設計清單（11 個）

| 軸 | 固定脈絡 | 子音 | 大小 |
|---|---|---|---|
| voicing × manner | place=front | p b f v | 2×2 |
| voicing × manner | place=middle | t d θ ð s z | 2×3 |
| voicing × manner | place=back | k g ʃ ʒ | 2×2 |
| voicing × place | manner=stop | p b t d k g | 2×3 |
| voicing × place | manner=short-fric | f v θ ð | 2×2 |
| voicing × place | manner=sibilant | s z ʃ ʒ | 2×2 |
| nasality × place | voiced, manner=stop/nasal | b m d n | 2×2 |
| manner × place | voiceless | p f t θ | 2×2 |
| manner × place | voiceless | t s k ʃ | 2×2 |
| manner × place | voiced | b v d ð | 2×2 |
| manner × place | voiced | d z g ʒ | 2×2 |

沒有的：voicing × nasality、nasality × manner（鼻音全有聲且全是 manner 0）；affrication × duration（企劃 A4 已說明）。

## Python d′（−6、0 dB）

d′ = z(H) − z(FA)，log-linear 校正；三水準軸給相鄰判準 (0∣1)、(1∣2)。「分另一軸」= 只用另一軸某水準的刺激，順序同該軸水準。
MRI p = 固定本軸刺激水準，本軸反應分布是否隨另一軸刺激水準而變（Pearson χ²，各水準合併）；n/a = 某反應水準從未出現。

| 子設計 | 子音 | 軸 (判準) | d′ −6 整體 | 分另一軸 −6 | d′ 0 整體 | 分另一軸 0 | MRI p (−6 / 0) |
|---|---|---|---|---|---|---|---|
| voicing×manner \| front | p b f v | voicing (0∣1) | 3.84 | 3.9/3.9 | 4.65 | 4.7/4.5 | 0.280 / 0.505 |
| voicing×manner \| front | p b f v | manner (0∣1) | 1.59 | 2.0/1.3 | 2.77 | 3.6/2.3 | 0.001 / <.001 |
| voicing×manner \| middle | t d θ ð s z | voicing (0∣1) | 3.54 | 4.5/2.7/3.9 | 6.22 | 5.6/5.4/5.6 | 0.001 / n/a |
| voicing×manner \| middle | t d θ ð s z | manner (0∣1) | 1.81 | 2.0/1.9 | 3.88 | 4.4/4.3 | <.001 / <.001 |
| voicing×manner \| middle | t d θ ð s z | manner (1∣2) | 1.09 | 1.1/1.1 | 1.11 | 2.3/0.4 | <.001 / <.001 |
| voicing×manner \| back | k g ʃ ʒ | voicing (0∣1) | 5.85 | 5.3/5.5 | 6.04 | 5.5/5.7 | n/a / n/a |
| voicing×manner \| back | k g ʃ ʒ | manner (0∣2) | 2.18 | 3.9/1.3 | 3.59 | 4.5/3.2 | <.001 / <.001 |
| voicing×place \| stop | p b t d k g | voicing (0∣1) | 5.24 | 4.6/5.1/5.6 | 6.42 | 5.8/5.7/5.8 | 0.367 / n/a |
| voicing×place \| stop | p b t d k g | place (0∣1) | 1.08 | 0.2/2.9 | 2.20 | 1.3/4.9 | <.001 / <.001 |
| voicing×place \| stop | p b t d k g | place (1∣2) | 0.43 | 0.5/0.4 | 1.24 | 1.2/1.3 | <.001 / <.001 |
| voicing×place \| short-fric | f v θ ð | voicing (0∣1) | 3.77 | 4.3/3.3 | 4.57 | 4.5/4.7 | 0.022 / 0.367 |
| voicing×place \| short-fric | f v θ ð | place (0∣1) | 0.50 | 0.5/0.5 | 1.33 | 1.0/2.1 | 0.673 / <.001 |
| voicing×place \| sibilant | s z ʃ ʒ | voicing (0∣1) | 4.60 | 3.9/5.6 | 6.05 | 5.5/5.7 | 0.008 / n/a |
| voicing×place \| sibilant | s z ʃ ʒ | place (1∣2) | 1.38 | 1.6/1.1 | 3.17 | 3.8/2.6 | 0.004 / <.001 |
| nasality×place \| voiced stop/nasal | b m d n | nasality (0∣1) | 3.55 | 3.6/3.4 | 5.76 | 5.3/5.7 | 0.769 / 0.274 |
| nasality×place \| voiced stop/nasal | b m d n | place (0∣1) | 1.93 | 2.8/1.6 | 4.05 | 4.9/3.7 | <.001 / 0.031 |
| manner×place \| voiceless | p f t θ | manner (0∣1) | 2.14 | 2.0/2.3 | 3.27 | 3.2/3.9 | 0.002 / <.001 |
| manner×place \| voiceless | p f t θ | place (0∣1) | 0.48 | 0.5/0.5 | 1.33 | 1.8/1.0 | <.001 / <.001 |
| manner×place \| voiceless | t s k ʃ | manner (0∣2) | 3.15 | 3.0/3.2 | 4.72 | 4.9/4.4 | 0.699 / 0.505 |
| manner×place \| voiceless | t s k ʃ | place (1∣2) | 0.94 | 0.4/1.5 | 2.51 | 1.8/3.5 | <.001 / <.001 |
| manner×place \| voiced | b v d ð | manner (0∣1) | 1.35 | 1.3/1.4 | 2.48 | 2.2/3.3 | 0.763 / <.001 |
| manner×place \| voiced | b v d ð | place (0∣1) | 1.09 | 2.2/0.4 | 2.73 | 4.1/1.9 | <.001 / <.001 |
| manner×place \| voiced | d z g ʒ | manner (0∣2) | 1.12 | 1.3/1.0 | 2.49 | 2.3/3.1 | 0.004 / <.001 |
| manner×place \| voiced | d z g ʒ | place (1∣2) | 0.54 | 0.3/0.7 | 1.56 | 1.2/2.3 | 0.023 / <.001 |

初步讀法（從上表推導，不是擬合結果）：nasality 與多數脈絡的 voicing 的 d′ 不隨另一軸變（可分性候選；例外：−6 dB 中位 voicing 在短擦音 2.7 vs 塞音 4.5，噝音 3.9 vs 5.6）；
place 的 d′ 強烈依 voicing／manner 而變（塞音 p∣t 在無聲 0.2、有聲 2.9 — M&N p. 347 說的清塞音 F2 轉折被送氣蓋掉）；
manner 在有聲比無聲差（1.3 vs 2.0、1.3 vs 3.9）。MRI 幾乎全顯著的那些是 place 與 manner，和上面一致。

## R 輸出看什麼

- `r_mdsdt_model_comparison.csv`：每個 (design, snr) 比 `full` 與 `PS_x`、`PS_y`、`PI_all` 的 AIC；
  PS 模型 AIC 不比 full 差 → 該軸知覺可分；`PI_all` 不差 → 知覺獨立。
- `r_mdsdt_results.csv` 的 `rho_stim*`：各刺激雲的相關；`mean_*`：四團雲的中心（看是否成矩形 = 可分）。
- `r_mdsdt_tests.csv`：mriTest（對照 Python 的 MRI χ²，應同向）、riTest。
- PDF：橢圓長軸是否傾斜（ρ ≠ 0）、四個中心是否排成矩形。
- 2×3 設計若 log 寫「無法擬合」，改看 Python 表，或拆成兩個 2×2（已有 `t s k ʃ`、`p f t θ` 這類）。

## 未核實

- 已對照 CRAN 文件（2026-11-10）：`fit.grt(freq, PS_x=FALSE, PS_y=FALSE, PI="none")`，`PI` 可為 `"none"`/`"same_rho"`/`"all"`；`mriTest`、`riTest` 存在；`silbert09a` 為 2×2 frequency × duration 混淆矩陣；模型比較用 `GOF`（AIC）。
- 仍憑記憶：回傳物件欄位（`means` / `covmat` / `a1`…）、`plot.grt`；R 腳本這些地方仍標 `[UNVERIFIED]`，各包 tryCatch。
- 4×4 矩陣版面假設為 x 跑最快 (x0y0, x1y0, x0y1, x1y1)；請對 `str(silbert09a)`，不對就改 `LAYOUT`。
- 2×3（兩軸水準數不同）mdsdt 能否吃 xtabs 形式：未知，腳本會先試 xtabs 再試矩陣，失敗只記 log。
- 本容器沒有 R，腳本沒跑過；Python 部分已執行、測試通過。
