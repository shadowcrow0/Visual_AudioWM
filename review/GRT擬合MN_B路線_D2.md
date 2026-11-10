# GRT 擬合 M&N：B 路線 D2 —— mdsdt 擬合結果與「客觀聲音尺」

2026-11-10。資料：Arc 上跑 `grt_mn_subdesigns.R`（mdsdt 1.3，R 4.x）的輸出
`review/data/grt_subdesigns/r_mdsdt_{model_comparison,results,tests}.csv`；
本檔的整理表存成 `mdsdt_summary.csv`（每個設計 × SNR 一列）與 `pair_dprime_mdsdt.csv`（每對子音一個擬合 d′）。
接續 `GRT擬合MN_B路線_D1.md`；目的（使用者 11-10 說明）：**找一把客觀的尺來排 SFT 的聲音刺激**，不是改 GRTv3_ada。

## 一句話

Voicing 與 nasality 是乾淨的一維軸：d′ 不隨另一個特徵變、與另一軸知覺獨立。
Place 和 manner 不是：同樣「差 place」，p–t 的 d′ 是 0.46、b–d 是 2.5；同樣「差 manner」，k–ʃ 是 3.4、g–ʒ 是 1.1。
所以聲音尺不能寫成「差哪個特徵 = 多遠」，要寫成**每一對子音自己的 d′**；而這張 d′ 表和我們已經在用的 Luce 距離 Spearman = .97。

```
   16×16 M&N (−12/−6/0/+6 dB)
        │ 切 11 個特徵對子設計（2×3 再切成相鄰 2×2 → 14 個 4×4）× 4 SNR = 56 次
        v
   mdsdt::fit.grt  full / PS_x / PS_y / PS_xy / PI_same_rho / PI_all / PS_xy+PI_all
        │
        ├─ ΔAIC(受限 − full)  ≤ 2 保留、> 10 拒絕   → 哪些特徵「可分」「獨立」
        └─ full 模型的均值差   = 該對子音的 d′       → pair_dprime_mdsdt.csv（客觀尺）
```

## 1. 跑了什麼、哪些沒跑成

- 56 個 (子設計, SNR) × 7 個模型 = 392 次擬合，336 次成功。失敗 56 次：25 次「共變異矩陣非正定」、其餘是模型在稀疏表上發散；幾乎都在 0/+6 dB 的 PS_y、PS_xy、PI_same_rho。−6 dB 只有 1 次失敗。
- mdsdt 的 2×2 路徑固定每個刺激雲的變異數 = 1，只估均值與 ρ；所以「可分性」在這裡 = 均值不隨另一軸移動。
- 每格先 +1 平滑；0/+6 dB 有一半以上的格子是 0，那裡的 ρ（−0.9 到 +0.9 亂跳）不可信。**結論以 −6 dB 為主、0 dB 為輔。**
- ⚠ mdsdt 1.3 的 `mriTest` 在 y 軸的 z 值公式分母寫反（`rB.sA1B/nA2B − rB.sA2B/nA1B`，原始碼 `grt_base.R` mriTest 內），列總數不等時會給假顯著；我們的子矩陣列總數差很多，所以 `r_mdsdt_tests.csv` 裡 `(-,B1)`/`(-,B2)` 那兩列全是 p = 0，**不要用**；x 軸的 `(A1,-)`/`(A2,-)` 正常。可分性一律看 ΔAIC（或 D1 的 Python MRI χ²）。

## 2. 可分性（ΔAIC = AIC(PS 模型) − AIC(full)；負值 = 加限制反而更好）

```
   設計 (子音)                軸 f 在軸 g 上可分？        軸 g 在軸 f 上可分？            知覺獨立 PI_all
                              −6      0      +6          −6      0      +6              −6     0     +6
   ───────────────────────── ─────── ────── ──────      ─────── ────── ──────          ────── ───── ─────
   voicing×manner  p b f v    −2.7   −3.8   −2.7  ✔     +8.4  +14.6   +9.0  ✘          +27    +31   +21  ✘
   voicing×manner  k g ʃ ʒ    −3.6   −3.6   −3.8  ✔      ×      ×     −1.4  ✔?         +1.9   +9    +17  −6 ✔
   voicing×manner  t d|θ ð    +7.2   −2.7   −1.5  ✔?    +4.4   −0.7    ×              +1.3  +11    +14  −6 ✔
   voicing×manner  θ ð|s z    +3.3   −3.8   −2.0  ✔     −1.4    ×    +12.4             −3.9   −3.4  +2.5  ✔
   voicing×place   p b|t d    −3.6   −3.9   −3.7  ✔      ×      ×      ×   (0.46 vs 2.8) ✘  +1.9   +6   +13  −6 ✔
   voicing×place   t d|k g    −3.9   −4.0   −3.8  ✔     −2.1   +2.8    ×   ✔           −7.7   −4.7  +2.1  ✔
   voicing×place   f v θ ð    +2.5   −3.6   −1.6  ✔     −3.2  +18.9   +3.8  ✘(0 dB)     +1.6   −0.5  −4.1  ✔
   voicing×place   s z ʃ ʒ    +2.3   −3.8   −3.9  ✔     +6.7  +10.5   −3.4  ✘          −4.9   +4.7  +15   −6 ✔
   nasality×place  b m d n    −3.6   −3.5   −3.9  ✔    +14.2   +0.6   −2.5  ✘(−6)       +5.0  +11   +17  ✘
   manner×place    p f t θ    +8.0  +20.3   +6.3  ✘    +12.8  +48.2    ×   ✘           −0.4  +19   +10  −6 ✔
   manner×place    t s k ʃ    −3.2   −3.5   −3.6  ✔    +42.5    ×     +6.0  ✘          +15   +18   +10  ✘
   manner×place    b v d ð    −3.6    ×    +21.6  ✘    +78.2    ×    +80.7  ✘           +32   +93   +44  ✘
   manner×place    d z g ʒ    +6.9  +31.8   −2.1  ✘     +3.5  +46.1  +52.2  ✘           +26   +39   +75  ✘
   × = 該模型沒擬合成功。「t d|θ ð」= 2×3 切出的相鄰 2×2。
```

讀法：
- **voicing**：在所有脈絡都可分（左欄一路負值；例外只有 −6 dB 的 θ–ð，因為 θ–ð 的 voicing d′ 本來就較低 2.5–3.1，拉低了平均）。而且在 −6 dB 與 place、manner 知覺獨立（PI_all ≤ 2），唯一例外是唇音 p b f v。
- **nasality**：可分；但 place 在鼻音/塞音間不可分（−6 dB +14.2）。
- **place**：幾乎處處不可分——無聲塞音 p–t 0.46 vs 有聲塞音 b–d 2.5–2.8（M&N p. 347 說的送氣遮住 F2 轉折）；噝音 s–ʃ 1.6 vs z–ʒ 0.9；對 manner 更是全部拒絕。
- **manner**：對 voicing 不可分（唇音 p–f 2.0 vs b–v 1.3；軟顎 k–ʃ 3.4 vs g–ʒ 1.1）；對 place 在 t s k ʃ 可分、其他拒絕。
- **manner × place 的雲是相關的**（PI_all 全拒，−6 dB 也是），voicing 的雲不是。

## 3. 每對子音的擬合 d′（客觀尺的第一版）

同一對出現在兩三個子設計時取平均；各設計間的範圍大多 ≤ 0.4（例外 d–ð 1.40/2.04、θ–ð 2.48/3.13、b–d 2.16/2.78）。

```
   −6 dB                                  0 dB
   對      特徵      d′       對      特徵      d′       對      特徵      d′
   ────── ───────── ────      ────── ───────── ────      ────── ───────── ────
   d–g    place     0.36      p–f    manner    1.97      ð–z    manner    0.73
   t–k    place     0.42      t–θ    manner    2.04      f–θ    place     1.00
   p–t    place     0.46      b–d    place     2.53      d–g    place     1.25
   f–θ    place     0.46      θ–ð    voicing   2.74      t–k    place     1.78
   v–ð    place     0.48      t–s    manner    2.88      p–t    place     1.81
   z–ʒ    place     0.90      d–n    nasality  3.24      v–ð    place     1.96
   θ–s    manner    0.97      k–ʃ    manner    3.36      θ–s    manner    2.20
   g–ʒ    manner    1.12      b–m    nasality  3.43      d–z    manner    2.21
   d–z    manner    1.25      s–z    voicing   3.63      b–v    manner    2.22
   b–v    manner    1.26      p–b    voicing   3.74      z–ʒ    place     2.38
   ð–z    manner    1.35      f–v    voicing   3.82      g–ʒ    manner    3.00
   s–ʃ    place     1.56      t–d    voicing   4.13      p–f    manner    3.24
   m–n    place     1.56      k–g    voicing   4.36      d–ð    manner    3.44
   d–ð    manner    1.72      ʃ–ʒ    voicing   4.61      s–ʃ    place     3.49
                                                          m–n    place     3.53
                                                          … voicing 對 4.2–4.8
```
（完整 56 列含脈絡見 `pair_dprime_mdsdt.csv`。）

**對照現用的 Luce 軸**（`consonant_distance.luce`，−12/−6/0 平均）：28 對單特徵對的 Spearman = **.974**（−6 dB）、**.973**（0 dB）。
也就是說 Luce 距離已經把順序排對了；GRT 多給的是（a）d′ 的單位（知覺噪音的 SD），（b）上面那張「哪些特徵能當軸」的結構。

## 4. 對 SFT 聲音尺的結論

1. **尺的單位用 d′（或等價的 Luce 距離），不用特徵數、不用 −log10(混淆次數)。** 特徵數的問題在 §2 一目了然：差 1 個 place 可以是 0.4 也可以是 2.5。
2. **只有 voicing 與 nasality 能當「軸」用**（d′ 隨脈絡穩定、與其他特徵獨立）。但它們的 d′ 全擠在 3.2–4.8，幾乎沒有「近」的一端；要「近」只能靠 place/manner 對，而那些對的 d′ 必須**按對查表**、不能按特徵給。→ SFT 的聲音「水準」本質上是**一張對子表**（現在的 `stimuli/sound_pairs.csv` 的做法是對的），不是一條連續軸；adaptive 走的是這張表的 d′ 欄。
3. **一個 SNR 下 d′ 的範圍只有 0.4–4.8**，再「遠」就沒有更遠的對了（voicing 對都在 4 上下）。這是知覺上的上限；11-03 在 WM 負荷下連這些對都在猜測率，所以瓶頸不在尺，在負荷（見 `pilot_2026-11-03_findings_EN.md`）。
4. 兩個可能要避開的對：θ–ð（voicing d′ 比其他 voicing 對低 1 以上，而且 θ/ð 對非母語者是另一個問題）、以及 0 dB 以上的任何 place 對（資料太稀，d′ 不穩）。

## 5. 下一步（D3）

```
   a  用 pair_dprime_mdsdt.csv（−6 dB）重生 sound_pairs.csv：
      五層的邊界照 d′ 切，每層的對挑「各設計一致」的（範圍 ≤ 0.5）           半天
   b  拿 11-03 的 data/yycc_sound_perception.* 對照：
      命中率 / RT 是否隨 d′ 單調——這是「噪音 d′ 能不能當我們的尺」的直接證據   半天
   c  決定 WM 負荷怎麼降（這才是 11-03 聲音在猜測率的原因），尺改好也救不了負荷  設計決策
```

企劃書裡的 A 路線（自寫五維 likelihood）暫不做：B 已回答「哪些特徵可分」，而 pair d′ 和 Luce 的 .97 表示再擬合一次不會改變順序。

## 6. 限制（寫進方法時要講）

- 來源是 M&N 1955：美式英語、CV + /ɑ/、白噪音、5 位說話者兼聽者。
- 子矩陣**條件化在子集內的反應**上（保留率 −6 dB 約 .55–.80）；d′ 是「在這幾個候選之間」的 d′。
- 2×2 路徑假設等變異數；均值差即 d′。
- 0/+6 dB 的擬合靠 +1 平滑撐著；ρ 不可信；PS 檢定在那裡有一半沒收斂。
- mdsdt `mriTest` 的 y 軸結果有 bug（§1），本檔沒有用它。
