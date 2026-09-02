# 決策脈絡 — SNR 噪音基準的單一參考點提案、與「是 bi / 不是 bi」的判斷形式

對話紀錄整理（2026-09-02）。四個問題依序：GRT 進度、AGRT 為何不能吃 SNR、
SNR 乘上什麼產生噪音、以及自然音 bi 當目標時判斷形式算不算完成。

---

## 1. GRT 進度（對話當下的 repo 狀態）

- `GRTv2.py:850-986`：AGRT 適應階段已完整接進主實驗——60 試次、兩維同時跑，
  `estimateGRTintensities(0.64)` 覆寫主實驗的兩個顏色與兩個聲音刺激
  （`GRTv2.py:959-973`）。
- 聲音維度已不是 SNR，是 Kutlu & McMurray (2024) 自然 b/p 連續體
  beachpeach step 1..9（`GRTv2.py:863-865`），端點即預設 AUDIO_B / AUDIO_P。
- 最新 commit `dea6fdf`（WITHOUT ADAPTIVE PART）加了 `GRTv3_demo.py`：
  拿掉適應階段的 demo 版，demo 無 SNR 操弄（`GRTv3_demo.py:271`）。

## 2. 「SNR 是連續的，為什麼 AGRT 不能用」——連續不是判準

AGRT 要的不只是連續軸，是「受試者直接對這條軸做低/高判斷、軸上存在主觀界線
alpha」的雙極維度。模型只有一行（`AGRT.py:133`）：

```
P(r=0 | x) = delta/2 + (1-delta) * Phi((alpha - x) / beta)
```

SNR 三處對不上（詳細論證見 `snr_vs_grt_dimension.md`，此處僅摘要）：

1. 反應變數：受試者答的是 b/p 類別，沒有人在判斷「吵不吵」。
2. SNR 動的是分母不是分子：

```
 語音連續體 step (x)  --->  移動刺激相對 alpha 的位置   <-- AGRT 搜尋的
                              |
              d' =  (x 到 alpha 的距離) / beta
                              |
 SNR (噪音量)        --->  放大 beta(有效知覺雜訊)     <-- 難度旋鈕
```

3. 輸出形狀：`estimateGRTintensities` 每維回傳 alpha 兩側對稱的 (L, H)
   兩個值；SNR 只需要一個值同套 b 和 p，兩個 SNR 會讓噪音量與音素身分共變。

「測量 SNR 影響度」的連續是另一種連續：測的是心理計量函數 accuracy vs
SNR dB（一條曲線、一個門檻），適合一維 Quest/Psi，不是 GRT 維度。
對比度、音量、時長也都連續，一樣是難度旋鈕。

## 3. SNR 乘上什麼產生噪音

兩層（皆為 `snr_audio.py` 字面內容）：

- **內容**：speech-shaped noise——頻譜整成 be/pe 語音自身 LTAS 的隨機噪音
  （`snr_audio.py:197-244`）；running noise，每試次新樣本、記種子供重建。
- **大小**：噪音位準 = 語音 RMS / 10^(SNR/20)（`snr_audio.py:268`）。

```
be/pe token ---> LTAS ---> 整形白噪音 ---> SSN (rms=1)
     |                                      |
     +--> rms(speech) --+                   v
                        +--> 縮放 = rms(speech)/10^(SNR/20)
     SNR dB ------------+                   |
                                            v
                              speech + noise --> 固定輸出 RMS 播放
```

## 4. 提案（未實作）：噪音基準改成單一參考點

**發現的不一致**（從程式碼推導，非文件記載）：

- 正規化對齊的是**有聲段** RMS（`_voiced_rms`，`snr_audio.py:120-141`，
  對到 `TARGET_RMS`，理由見 `snr_audio.py:43-46`）。
- 但混音定噪音量用的是**全檔** RMS（`snr_audio.py:268` 的 `_rms(x)`）。

pe 的送氣/靜音比例比 be 高，有聲段對齊後全檔 RMS 仍不同 → 同一名目 SNR 下
be/pe 試次的噪音絕對位準有系統性小差，噪音量與音素身分共變。噪音又比語音
早 200 ms 起來（`snr_audio.py:59-60`），引導段位準原則上是在語音出現前
洩漏身分的線索。

**提案**：

```
現況:  noise_rms = rms(x_token) / 10^(SNR/20)    <- 隨 token 變
提案:  noise_rms = REF_RMS      / 10^(SNR/20)    <- 只隨 SNR 變
```

REF 建議直接用 `TARGET_RMS` 設計常數（有聲段已對齊到它，等價於「相對 be」
但不必單挑一個 token 當基準）。附帶兩點：

1. `snr_audio.py:273` 的輸出增益分母 `_rms(sp + noise)` 也隨 token 變，
   要讓耳朵端噪音位準真正只由 SNR 決定，這步也得換成不依賴 token 的固定配方。
2. 「SNR 代表噪音量」指混音配方內的相對位準；最終播放位準仍被 `OUTPUT_RMS`
   固定（`snr_audio.py:53-58`），這是刻意的（避免音量成為難度線索與削波）。

## 5. 自然音 bi 與「是 bi / 不是 bi」的判斷形式

自然 bi 已在 repo：`stimuli/kutlu_mcmurray_2024/cv/beachpeach1_cv.wav`
（"beach" 的 CV 段 = 自然 /bi/；step 9 = /pi/）。

「是 bi / 不是 bi」——**弱意義上有，嚴格意義上沒有**：

- 有：刺激集只有兩個時，「b 還是 p」（`GRTv2.py:928-930`）與
  「是 bi / 不是 bi」在資料上完全等價，只是按鍵標籤不同。
- 沒有，三點：
  1. 「不是 bi」被操作化成只有 pi 一個對照，推論範圍是 bi vs pi，
     不是 bi vs 任何非 bi（見 `review/exemplar與類別的推論範圍.md`）。
  2. GRT 要的本來就是 2x2 辨識反應填 4x4 混淆矩陣，不是 yes/no 偵測
     （偵測有反應標準偏誤問題）；維持二選一辨識是設計正確，非缺件。
  3. 適應階段的中間 step 在量類別界線，不是在測 bi 偵測率
     （step 5 無正解不給回饋，`GRTv2.py:936-938`）。

```
校準階段 (已實作)          主實驗 GRT          若 WM 任務要「目標偵測」
bi <---連續體---> pi       2x2 刺激            「這是不是剛剛那個 bi」
受試者答 b/p               4x4 混淆矩陣         = 另一層任務設計,
= 「是bi/不是bi」的等價形式  (辨識,非偵測)        尚未實作
```

---

分層標示：第 1、3 節為程式碼字面內容；第 2 節的模型要求為既有文件
（`snr_vs_grt_dimension.md`、`review/決策脈絡_AGRT模型假設.md`）之摘要；
第 4 節的不一致是本次對話從程式碼推導的，提案未實作；第 5 節的等價性論證
是推論，引用行號皆已比對實際檔案。
