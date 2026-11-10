# 企劃：用 GRT 擬合 Miller & Nicely (1955) 的子音混淆矩陣

草稿 2026-11-10。目的：把 16 個子音之間的「距離」用 d′ 定義出來（GRT 的語言），同時正式檢定五個構音特徵是不是獨立的知覺維度，取代現在用 Luce η 排序的做法。資料與程式都在本 repo。

## 1. 要回答的三個問題

```
   Q1  每個特徵（voicing / nasality / affrication / duration / place）在每個 SNR 的 d′ 是多少？
   Q2  五個特徵是不是五條互不干擾的知覺軸？（perceptual separability / independence、decisional separability）
   Q3  用 d′ 算出來的子音間距離，和 Luce η、資訊傳遞量、純特徵數排序差多少？對刺激選擇有沒有影響？
```

## 2. 資料

- `review/data/miller_nicely_1955_tables_I-VI.json`：六張 16×16 噪音混淆矩陣（S/N −18 … +12 dB，每列 ≈ 250 次、每張 4000 次），已對過列和與論文 Table XXI（誤差 ≤ .05）。
- 主用 −12、−6、0、+6 dB。−18 dB 近乎均勻（整張矩陣 0.06 bit），+12 dB 240 個非對角格有 199 個是 0，估不出什麼。
- 設計性質：這是 **16 刺激完全辨識**（complete identification）設計，不是常見的 2×2。優點是每個特徵有很多對刺激可以互相約束；缺點是標準 GRT 工具（`grtools`、`mdsdt`）只處理 2×2 或 2×2×…，要自己寫 likelihood。

## 3. 模型：一層一層放寬

每個子音 i 在五維知覺空間裡是一團多變量常態雲；受試者把空間切成 16 個反應區，P(答 j | 聽 i) = i 的雲落在 j 的區域的機率。全部用多項式概似估。

```
   M0  獨立通道（= M&N 的「五條通道」正式化）
       每個特徵 f 一條軸，刺激在軸上的位置 = 特徵值 × d′_f，變異 = 1，軸之間不相關，
       決策線切在兩值中間（place 三值：兩條線）。
       → 反應機率可分解：P(j | i) = ∏_f P(j 在 f 上的值 | i 在 f 上的值)
       參數：每個 SNR 5 個 d′（place 兩段可分開 → 6 個）。
   M1  M0 + 決策偏誤：每個特徵一個判準位移 c_f（偏好答無聲／答非鼻音…）。Phatak et al. 2008 說擦音 voicing 混淆不對稱，
       這一層就能看到。+5 參數。
   M2  M1 + 知覺可分性失敗：同一特徵的 d′ 依另一特徵的值而異（例：voicing 的 d′ 在塞音和擦音不同；
       place 的 d′ 在有聲 / 無聲不同）。這是 M&N 自己注意到的（p. 347：清塞音的 F2 轉折藏在送氣裡）。
       每個特徵對 (f, g) 一個交互項，先只放理論上有理由的三對。
   M3  M2 + 知覺獨立性失敗：雲在兩條軸上有相關 ρ_fg（同一刺激內，一個特徵聽偏了另一個也跟著偏）。
       16 刺激設計下 ρ 比 2×2 設計好認（Silbert & Thomas 2013 指出 2×2 裡 ρ 和決策線斜率會混）。
   M4  M3 + 反應層偏誤：每個反應 j 一個 b_j（對照 Luce 模型的 b）。做到這裡就能和 Luce 直接比。
```

模型比較用 G²（對飽和模型）、AIC、BIC，每個 SNR 各算一次，再看參數隨 SNR 的走勢（d′ 應單調上升；ρ、c 應穩定）。

## 4. 實作

```
   Visual_AudioWM/grt_mn.py
     features()          Table XIX 的 0/1/2 編碼（已在 adaptive_vawm_EN.MN_FEATURES）
     predict(theta)      給參數 → 16×16 預測機率矩陣
                         M0–M2：逐特徵 Φ 相乘（closed form）
                         M3–M4：五維常態在長方體區域的積分 → scipy.stats.multivariate_normal.cdf 的差分
                                （五維、16×16、每次評估 256 格，可接受；或用 Genz 演算法）
     nll(theta, N)       多項式負對數概似，加 0.5 平滑只用在 G² 的飽和模型
     fit(model, snr)     scipy.optimize.minimize（L-BFGS-B），多起點；d′ ≥ 0、|ρ| < 1
     compare()           G² / AIC / BIC 表；參數 × SNR 表
     dprime_matrix()     任兩子音的 Mahalanobis 距離 √((μ_a − μ_b)ᵀ Σ⁻¹ (μ_a − μ_b)) → 120 對
   review/data/grt_mn_fits.json、grt_mn_summary.txt
   tests/test_grt_mn.py  M0 用模擬資料回收 d′；M0 的預測在 M&N −6 dB 上 G² 要比 Luce 閉式解差不到一個數量級
```

依賴：numpy、scipy（已在 venv；PsychoPy 的 Python 不需要跑這個）。可以借 `adaptiveSFT` 的 Arc 流程跑多起點。

## 5. 驗證與對照

1. **內部**：M0 的 d′_f 應和 M&N 的資訊傳遞量單調對應（Table XXI）；用模擬資料回收。
2. **對 Luce**：M4 的 120 對 Mahalanobis 距離 vs `consonant_distance.luce` 的 −ln η：Spearman 預期 > .9（兩者都在描述同一張表）；差異集中在哪些對，寫下來。
3. **對我們的資料**：用擬合出的 d′ 預測 `sound_perception_EN.py`（`stimuli/sound_pairs.csv` 的 80 對）的命中率排序，和真人結果比。這是從「噪音裡的 d′」到「我們的作業」的第一個外部檢驗；如果排序對不上，就是噪音 ≠ 我們的情境的直接證據。
4. **GRT 線的回饋**：Q2 的結果（哪些特徵之間可分、哪些不可分）直接決定 GRTv3_ada 要不要把聲音維度換成「可分的那一對特徵」。

## 6. 產出

- `review/GRT擬合MN_結果.md`（中）+ `grt_mn_results_EN.md`（英）：d′ × SNR 表、模型比較表、可分性檢定結論、120 對距離、與 Luce 的差異清單。
- 一張圖：五個特徵的 d′ 隨 SNR 的曲線（對照 M&N Fig. 1）。
- 若 M2/M3 顯著：一段「哪些特徵互相干擾」的白話說明，給老闆。

## 7. 時程（實際工作天）

```
   D1   M0 + M1 寫好、跑四個 SNR、回收測試                  半天
   D2   M2 + M3 的積分與擬合（多起點、收斂檢查）             一天
   D3   M4、模型比較、對 Luce、對 sound_perception 資料       半天
   D4   寫結果、圖、英文版                                   半天
```

## 8. 風險與限制

- **資料稀疏**：+6 dB 以上非對角格大多為 0，M3/M4 在那裡會不穩；主要結論放在 −12 … 0 dB。
- **可辨識性**：完全辨識設計下 ρ 與決策線不像 2×2 那麼糾纏，但 M3 仍要用多起點和 profile likelihood 確認。
- **place 三值**：兩條決策線、兩段 d′；前/中/後是否等距不假設。
- **只是噪音裡的 d′**：和 Luce 一樣，這不是記憶裡的距離、也不是非母語者的距離（`millerNicely1955_噪音混淆_逐SNR.md` §7–8）。第 5 節第 3 點就是在量這個落差。
- **知覺雜訊 vs 刺激雜訊**：M&N 的雲寬度是「噪音 + 腦內雜訊」的和，GRT 分不開（Ashby 2000；Ashby & Wenger 手冊章節；`90_Sources/ashby2000.md`）。對排序沒影響，對 d′ 的絕對值解讀要小心。

## 9. 參考（repo 裡已有卡片者標 ✔）

- Ashby, F. G., & Townsend, J. T. (1986). Varieties of perceptual independence. *Psychological Review, 93*, 154–179.
- Ashby, F. G., & Perrin, N. A. (1988). Toward a unified theory of similarity and recognition. *Psychological Review, 95*, 124–150.（書目憑記憶）
- Ashby, F. G. (2000). ✔ `90_Sources/ashby2000.md`；Ashby & Wenger 手冊章節 ✔。
- Silbert, N. H., & Thomas, R. D. (2013). Decisional separability, model identification, and statistical inference in the general recognition theory framework. *Psychonomic Bulletin & Review, 20*, 1–20.（書目憑記憶）
- Silbert (2012, 2014) ✔；Silbert & Hawkins (2016) *JMP* 73, 94–109 ✔（GRT 教學）。
- Soto et al. (2015) GRT-wIND ✔；Soto et al. (2017) `grtools` ✔。
- Miller & Nicely (1955) 全文已讀；Luce (1963)、Shepard (1957, 1987)、Nosofsky (1986) 見 `consonant_distance_how.md`。
