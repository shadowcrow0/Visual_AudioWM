# Cutler, Weber, Smits & Cooper (2004) 全文查核：母語 vs 非母語聽者的差別在哪

*JASA, 116*(6), 3668–3678. https://doi.org/10.1121/1.1810292
2026-11-10 依使用者上傳的 PDF 全文查核（`papers/Cutler_etal_2004_JASA.pdf`）。頁碼為期刊頁碼。
修正先前筆記：「非母語者 voicing 傳得最差」**在這篇裡是真的**（p. 3674），但它是荷蘭語特有的詞尾無濁音對比造成的，不是「非母語者」的通性。

## 一句話

非母語者（荷蘭人聽英語）每個 SNR 都比母語者差一截、差距固定約 20%，噪音加重時差距**不**擴大；
難的音素兩群一樣難（逐音素正確率相關 .83–.91）；唯一清楚的母語效應是**詞尾的 voicing**——荷語詞尾沒有 b/d/g 對 p/t/k，荷蘭聽者在詞尾把 voicing 聽丟。

```
   % 正確                 0 dB     8 dB    16 dB
   ──────────────────── ──────── ─────── ───────
   母語（美式英語）16 人     ~56→     ~69→    ~75      整體 73.6%
   非母語（荷蘭）16 人       低約 20%，三個 SNR 都一樣      整體 60.0%
   → 群組 × SNR 不交互（p. 3670）；非母語 ≈ 母語的 80%（p. 3676）
```

## 設計（p. 3669–3670）

- 16 位美式英語母語者（南佛羅里達大學）、16 位荷蘭學生（奈梅亨；11 歲起學英語 7–8 年，流利，未長住英語國家）。
- 24 個子音 × 15 個母音，所有合法的 CV 與 VC = 645 個音節，一位美國中西部女性說話者。
- 多人交談噪音（babble），SNR 0 / 8 / 16 dB；每人 3870 試驗，分「報子音」與「報母音」兩種區塊。
- 噪音為什麼選 babble：模擬非母語者日常最難的情境（p. 3669）。

## 五個差異

### 1. 整體：差一截，但差距不隨噪音變

> "overall performance was better for native (grand mean 73.6% correct) than for non-native listeners (grand mean 60%; F[1,30]=21.1, p<0.001) ... the native versus non-native comparison interacted neither with SNR nor with the vowel/consonant factor." (p. 3670)

> "Non-native identification scores were around 80% of native identification scores at each of the SNRs used in the present study." (p. 3676)

所以「非母語者在噪音裡特別吃虧」不是音素辨識層的事；作者歸因於每一層處理都慢一點、差一點，累積起來（p. 3676）。

### 2. 難的音素兩群一樣難

> "the per cent correct identification rate of the two listener groups across phonemes was very highly correlated: at 16 dB SNR r=0.83, at 8 dB 0.87 and at 0 dB 0.91, in all cases p<0.001." (p. 3671)

英語有、荷語沒有的子音（θ ð ʒ dʒ、詞尾的 path/smooth/edge/egg）荷蘭人難，美國人也難；母音 bat–bet、cot–cut 兩群都混 (p. 3671)。

### 3. 特徵層：母語者 manner ≈ voicing > place；非母語者 manner > place > **voicing 最差**

> "for native listeners consonantal manner information was transmitted most accurately and place information least accurately; manner and voicing did not differ significantly but each was significantly more accurately perceived than place (p<0.001 for manner, p<0.01 for voicing). For non-native listeners manner was also transmitted most accurately (significantly more so than place, p<0.001), but voicing least accurately (significantly less so than place, p<0.001)." (p. 3674)

```
   資訊傳遞量排序          母語                  非母語（荷蘭）
   ─────────────────── ─────────────────── ─────────────────────
   子音                 manner ≈ voicing     manner
                             > place           > place
                                               > voicing   ← 反過來
   母音（兩群相同）      backness > height > tenseness
```
（特徵定義 Table IX, p. 3675：manner 6 值、place 6 值、voicing 2 值。）

### 4. voicing 掉在哪：詞尾

> "For the native listeners, voicing information was perceived better in final position, but for the Dutch listeners, as expected, voice was much less well perceived in final position [Fig. 3(e)]." (p. 3673)

> "The one clear effect of native phonology on non-native listening appeared in the Dutch listeners responses to syllable-final consonants; as described above, Dutch phonotactics prohibit voicing contrasts in final position, and the Dutch listeners made many more voicing errors on final consonants such as /b, d, g/ than native listeners did." (p. 3671)

> "the absence of a voicing contrast in final position in Dutch was reflected in the non-native listeners' poor performance on voicing decisions in final position [Fig. 3(e)], which led to a reversal, for non-native consonant identifications only, of the otherwise constant advantage of final over initial position" (p. 3675)

0 dB 混淆矩陣裡看得到（Table II vs VI，詞尾；Table I vs V，詞首）：

```
   詞尾 0 dB      母語 正確   荷蘭 正確   荷蘭最大的錯
   ───────────── ────────── ────────── ─────────────────
   p (liP)         50.0       24.2      → b  21.3%
   t (hoT)         77.1       45.0      → d  16.3%
   k (siCK)        63.3       44.6      → g  12.5%
   v (loVE)        47.5       15.8      → f  12.9%、→ ð 15.4%
   詞首 0 dB
   p (Pie)         15.4       30.8      （荷蘭反而較好；兩群都大量聽成 h）
   g (Go)          29.6       17.1
   v (Very)        17.5        9.6
```
而且這個詞尾 voicing 的劣勢**在噪音較輕時反而更明顯**（p. 3675：16 dB 時詞首優於詞尾的差距比重噪音下更大）——它是音韻知識的問題，不是訊號被蓋掉。

### 5. 位置：詞首比詞尾難（兩群都是），但荷蘭人的子音因為 voicing 反過來

> "for native listeners, identification of both consonants and vowels is better in final than in initial position. Non-native listeners show the same final-position advantage for vowels; for consonants, however, their performance is worse in VC than in CV." (p. 3670–3671)

## 對本研究的意義

1. 先前 §8 (d)「非母語者 voicing 最弱——未經查證」改為：**Cutler 2004 證實荷蘭聽者 voicing 傳遞最差，機制是 L1 詞尾無濁音對比**。這支持的是「特徵的流失跟著 L1 音系的缺口走」，不是「非母語者普遍 voicing 弱」。
2. 我們的受試者是華語母語，華語塞音對立是送氣而非濁音，而且**任何位置**都沒有英語式的 voicing 對比——依 Cutler 的邏輯，預測華語聽者的 voicing 劣勢會出現在詞首（我們用的是 CV）。但這是推論；Flege (1989)、Yang et al. (2026) 的摘要說華語聽者某些 voicing 對比近母語。仍需查全文或自己測。
3. 「噪音不會放大非母語劣勢」：若之後要用噪音調難度（SNR 軸），Cutler 的結果說母語與非母語的 d′ 會**平行下降**，不會交叉——對設計是好消息。
4. 兩群難的音素相同（r ≥ .83）：M&N 的混淆結構（母語者）拿來排華語聽者的刺激順序，方向大致可信；絕對值要打八折。
