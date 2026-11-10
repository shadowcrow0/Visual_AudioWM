# Roodenrys, Miller, Guitard & Neath (2025) 全文查核

*Memory, 33*(2), 248–258. https://doi.org/10.1080/09658211.2024.2433049
2026-11-10 依使用者上傳的 PDF 全文（12 頁，含附錄）查核；取代先前只到摘要層級的描述。
下面的引文為原文逐字，頁碼為期刊頁碼。

## 一句話

他們**沒有算距離**。「相似」只有一個定義：兩個子音**恰好差一個構音特徵**（voice、place 或 manner 三選一），
其餘一律歸為「不相似」；三個特徵沒有分開分析，60 個對裡 36 個是 voicing 對。

```
   target  "beg"  /bɛg/
                     ┌─ overlap   : 位置 2 "ban" 共用 /b/、位置 4 "wig" 共用 /g/   （同一音素）
   list 位置 2、4 ── ┼─ minpair   : 位置 2 "pan" /p/、位置 4 "wick" /k/           （各差 1 個特徵）
                     └─ control   : 無共用音素、也不是 minimal pair
   回憶目標字（位置 5）的正確率：overlap < minpair < control
```

## 六個問題

### 1. 用哪些特徵、來源

區辨特徵（distinctive features）框架，引 Chomsky & Halle (1968) 與 Ladefoged (1971)；只用在**子音**，母音不動。

> "The definition of phonemes in terms of a unique combination of articulatory features has a relatively long history (e.g., Chomsky & Halle, 1968; Ladefoged, 1971). These are often referred to as distinctive features and include such things as whether the vocal cords vibrate or not, the part of the mouth that is most open, and, if the vocal tract is fully closed during production, where that closure occurs." (p. 249)

> "For the purposes of this study, a definition based on articulatory features is adopted and applied only to the consonants of the words as this type of coding system provides especially clear examples of similar phonemes." (p. 249)

實際出現的特徵類型只有三種（附錄 Table A2, p. 258，欄名 "Type of feature difference"）：**Voice、Place、Manner**。沒有 nasality、沒有 affrication/duration 這種 M&N 式的細分。

### 2. 相似度怎麼算

不是量尺，是二元定義：差**恰好一個**特徵就算 minimal pair。

> "Using this system of classification, it is possible to identify phonemes that differ on a single feature. The term 'minimal pair' is usually used to refer to two words that differ on a single phoneme, but it is used in this paper to refer to two phonemes that differ on a single feature." (p. 249)

> "In the minimal pair condition, the initial consonant of the word in position 2 and the final consonant of the word in position 4, differed from the consonants in the target by a single articulatory feature. For example, 'pan' (pæn) and 'wick' (wɪk) for the target 'beg' (bɛg)." (p. 250)

沒有「差 2 個特徵」的條件、沒有共享特徵數的連續變項、沒有任何相似度指數。
control 的定義是「沒有共用音素且不是 minimal pair」(p. 250)，所以 control 可能差 2–4 個特徵，但文中沒算。

### 3. 刺激

- 30 個 CVC 目標字，固定在 6 字串列的**位置 5**；干擾字在位置 2 和 4；位置 1、3、6 是固定 filler (p. 250)。
- 三個條件的目標字共 60 個子音；Table A2 (p. 258) 逐對列出 minimal pair 子音與次數：

```
   對       次數   特徵類型            對       次數   特徵類型
   ──────── ────   ────────            ──────── ────   ────────
   k/g       12    Voice               l/r        5    Place
   p/b       10    Voice               d/b        4    Place
   d/t        6    Voice               p/t        3    Place
   tʃ/dʒ      4    Voice               n/m        7    Place
   v/f        3    Voice               θ/t        3    Manner
   z/s        1    Voice               θ/s        2    Manner
            ────                                ────
   Voice     36 (60%)          Place   19 (32%)      Manner   5 (8%)
```

> "For 36 of the 60 consonants of the targets, the minimal pair consonant differed on voicing, meaning that they are produced with the same mouth position/movements except that in one phoneme the vocal cords vibrate and the other they do not, e.g., the 'b' and 'p' or 'k' and 'g' (hard g) phonemes" (p. 250)

- 位置 2 的字在三條件間以 SUBTLEX 詞頻與字母數配對；位置 4 只配到詞頻，control 的字平均長 0.4–0.5 個字母 (p. 250)。
- Exp 2 的錄音：男性澳洲英語說話者，Røde Broadcaster 麥克風，Audacity 2.1.1，振幅正規化 (p. 252)。
- 資料與刺激在 OSF (p. 256)。

### 4. 作業

```
   Exp 1   線上 · 視覺呈現 1 字/秒 · 打字序列回憶 · 60 人 · 2 練習 + 30 正式
   Exp 2   實驗室 · 聽覺呈現 1 字/秒 · 口說序列回憶 · 60 人 · 同一組 30 串列
```

> "They were then presented with six words in white on a black background, one after the other, in the centre of the screen at a rate of one word per second. ... participants typed their response before clicking another button to submit the response and move on to the next word." (p. 250)

> "The second experiment is a replication of the first, using the same stimuli, with the exception that they were presented auditorily and recall was spoken." (p. 252)

依變項：目標字（位置 5）的 correct-in-position 與 item recall；只做 directional paired t-test + Bayes factor (p. 251)。

結果（目標字回憶比例）：

```
                     control   minpair   overlap    minpair vs control
   Exp 1  位置正確     .308      .258      .237      d = .26, BF₁₀ = 1.8  (anecdotal)
   Exp 1  item        .478      .417      .374      d = .30, BF₁₀ = 3.4  (moderate)
   Exp 2  位置正確     .342      .288      .187      d = .31, BF₁₀ = 4.0  (moderate)
   Exp 2  item        .483      .398      .295      d = .39, BF₁₀ = 17.2 (strong)
```
(p. 251, p. 252–253)。Exp 2 中 overlap 又顯著差於 minpair（d = .50–.60, p. 252–253）。

### 5. 哪些特徵重要

**沒有分析。** 三種特徵類型從未被分開比較，也沒有報告 voicing 對 vs place 對的干擾量。
文中唯一接近的句子是在談**視覺**相似時說配對無法做統計：

> "The stimuli pairs mostly involve one similar and one dissimilar letter, making it impossible to run statistical comparisons, and a visual examination of the data does not suggest a clear pattern." (p. 255)

所以能從這篇得到的只是「差一個特徵的子音仍會干擾」，且這個結論有 60% 的權重來自 voicing 對。

### 6. 跟「聽覺混淆」的關係（對本研究最要緊的一段）

他們**明確把構音相似與聲學混淆分開**，並用 Wang & Bilger (1973) 的混淆資料說只有約一半的 minimal pair 在聲學上會混：

> "The similarity was defined in terms of the distinctive articulatory features (Chomsky & Halle, 1968) of the phonemes rather than acoustic similarity. Phonemes that have similar articulatory features can be acoustically similar but this is not always true. Only about half of the minimal pair phonemes in the stimuli set are acoustically confusable (Wang & Bilger, 1973). However, if the effect was due to acoustic confusability it would be reasonable to expect it to be absent from Experiment 1 which did not use auditory presentation of the stimuli. The extremely similar results of the two experiments argues against the effect being driven by acoustic similarity." (p. 254)

理論主張：回憶時的反應抑制（response suppression）沿著構音特徵擴散，而非回憶時的覆寫 (p. 254–255)；並引 Newman & Dell (1978) 的音素監控結果（目標 /b/ 前面出現 /p/ 時偵測最慢、其他塞音次之、非塞音最快）當作「特徵層級有漸層效應」的旁證 (p. 255)。

## 對本研究的意義

1. **記憶文獻的「軸」表要改**：Roodenrys 2025 用的是 voice / place / manner **三軸、二元（差 1 特徵或否）**，不是連續距離，也不是 M&N 五軸。
2. **(c)「記憶混淆結構＝知覺混淆結構」要往下修**：這篇是反方證據——干擾在視覺呈現＋打字回憶下一樣出現，而且作者引 Wang & Bilger 說一半的 minimal pair 聲學上不混。它支持的是「構音特徵相似」而不是「噪音中的聲學混淆」。
3. **它不能證明「特徵距離 1 到 5 有漸層」**：只有 0 / 1 / 其他三格，而且第三格沒有定義距離。我們若要主張漸層，證據只能來自 M&N 的知覺資料與 Newman & Dell 1978（音素監控，非記憶）。
4. **voicing 在記憶中不是「弱」**：這篇 60% 的干擾對是 voicing 對，且干擾顯著；與先前查核結論一致——不能說非母語者或記憶系統丟掉 voicing。

## 另見

Schweppe, Grice & Rummer (2011) 全文查核：`Schweppe2011_全文查核.md`。
