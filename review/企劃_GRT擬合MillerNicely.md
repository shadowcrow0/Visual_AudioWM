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

**軸本身也要比**（五條軸是假設進去的，不是擬合出來的；Silbert & Hawkins 2016）：

```
   A5  M&N 的五軸（上面的 M0–M4）
   A4  affrication 與 duration 合併成一條三階 manner 軸：塞音/鼻音 → 短擦音 (f θ v ð) → 噝音 (s ʃ z ʒ)。
       這不是為了省參數，是結構上必然：沒有「不摩擦但長」的子音，(affrication 0, duration 1) 那格是空的，
       duration 巢在 affrication 底下，兩者在資料裡分不開（權重 .44/.44、M&N 自承 duration 名字是任意取的 p. 346）。
       所以「affrication × duration 可不可分」用真實子音做不成 2×2；能做的是 manner × voicing、manner × place（4b 節）。
   A3  voicing、nasality、一條「manner + place」的合併軸（對照 MDS 三維已能解釋 .95）
   AK  Klatt (1968) 的特徵組（long frication、continuant、sonorant、sibilant、voiced）
```

每組軸各跑 M0–M1，用 BIC 比；若 A4 或 A3 不比 A5 差，之後的 M2–M4 就用較少的軸（參數少、ρ 更好認）。

**記憶文獻用的是幾個軸**（2026-11-10 查核；除 Roodenrys 2025 與 Schweppe 2011 已讀全文外皆摘要／片段層級；細節見 `millerNicely1955_噪音混淆_逐SNR.md` §7–8、`Roodenrys2025_全文查核.md`、`Schweppe2011_全文查核.md`）：

```
   研究                          作業                      用幾個軸、哪些軸
   ──────────────────────────── ───────────────────────── ────────────────────────────────────────────────
   Wickelgren 1966               23 子音 CV 序列回憶        4：voicing、nasality、openness（manner）、place（5 值）；三套系統中最準
   Klatt 1968                    重分析 Wickelgren 矩陣     5 個二元：long frication、continuant、sonorant、sibilant、voiced（無 place）
   Hintzman 1967                 無意義音節 STM             2：voicing、place
   Cole, Haber & Sales 1973      聽覺編碼、記憶混淆         Chomsky–Halle 區辨特徵組
   Cole / Sales 1974             序列回憶                   manner vs place；place「沒有被記住」
   Danhauer et al. 1978          Wickelgren 作業 + 噪音     INDSCAL 抽維度（維數未見）
   Schweppe, Grice & Rummer 2011 序列回憶（聽/視呈現 × 書寫/口說） 2 類：MANNER（pa ta ka）vs PLACE 主動構音器（da la za）；不算特徵數；共享 manner 任何模式都傷、共享 place 只口說時傷（全文已讀）
   Roodenrys et al. 2025         序列回憶（視覺+打字／聽覺+口說） 3：voice、place、manner；二元「差 1 特徵」，36/60 對是 voicing，特徵未分開分析（全文已讀）
   Pisoni 1973                   延遲 same/different        1：ba–pa voicing 連續體
```

主流是 Wickelgren 的四軸 = M&N 五軸把 affrication 與 duration 併成 manner，即上面的 **A4**。所以 A5 vs A4 的比較就是「知覺的軸 vs 記憶的軸」；Klatt 的特徵組（AK）提醒 place 可能不在記憶的軸裡。

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

## 4b. B 路線：切成特徵對的子設計，用現成套件

`mdsdt`（Silbert & Hawkins 2016）的 `fit.grt` 處理**兩個知覺維度、每維 N 個水準**的設計：標準 2×2 之外，可用 `xtabs`（刺激 × N×N 反應）餵 3×3（內建 `thomas15a/b` 範例）。它假設 decisional separability，檢定 perceptual separability 與 independence。它**不能**直接吃 16×16：模型是二維常態，而 16 個子音是五條軸（2×2×2×2×3）。`grtools`（Soto et al. 2017）只有 2×2。（以上依 2026-11-10 搜尋到的說明頁片段；rdrr.io 與 CRAN 在雲端 session 被擋，原始碼未核對——`?fit.grt`、`str(thomas15a)` 在筆電上一看即知；2×3 這種兩軸水準數不同的設計能不能跑，未核實。）

用法：固定其他特徵，切出兩個特徵的子矩陣，各跑一次：

```
   voicing × place（塞音）        {p t k b d g}          6×6   2×3
   voicing × place（擦音，長）    {s ʃ z ʒ}              4×4   2×2
   voicing × place（擦音，短）    {f θ v ð}              4×4   2×2
   voicing × affrication（前）    {p b f v}              4×4   2×2
   voicing × affrication（中）    {t d θ ð}              4×4   2×2
   nasality × place              {b d m n}              4×4   2×2
   duration × place（有聲擦音）   {v ð z ʒ}              4×4   2×2
   …（每對特徵在每個可切出的子集各一次，每個 SNR 各一次）
```

子矩陣的列和是原列和的一部分（只保留落在子集內的反應），要在報告裡說明是「條件化在子集上」的分析。

```
   A 路線  自己寫五維 likelihood（第 4 節）            完整；要寫積分、確認參數可辨識
   B 路線  特徵對子矩陣 + mdsdt / grtools              現成、一天內有結果、每對特徵一個結論；看不到三個以上特徵的交互
   順序    先 B；B 指出哪幾對特徵不可分，A 只針對那幾個特徵做
```

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
